import os
import json
import logging
import random
from typing import Tuple, Optional

import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retrieval_pipeline import UniversityIR
from src.preprocessing import turkish_preprocess
from src.tuning import (
    tune_alpha,
    tune_feature_weights,
    learn_alpha_by_type,
)
from src.qrels_utils import load_qrels_json, build_qrels_mapping, extract_queries
from src.save_ir import save_university_ir

from src.train_custom_cross_encoder import (
    train_cross_encoder_with_hard_negatives
)


# ==========================================================
# CONFIG
# ==========================================================
class Config:
    CORPUS_PATH = "data/processed/corpus_preprocessed.json"
    QREL_TRAIN_PATH = "data/qrel_train_scored_final.json"

    DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    CROSS_ENCODER_BASE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Retrieval defaults
    DEFAULT_ALPHA = 0.5
    DEFAULT_ALPHA_BY_TYPE = {
        "date_query": 0.7,
        "short_query": 0.6,
        "default": 0.4,
    }

    DEFAULT_WEIGHTS = {
        "expansion": 0.4,
        "prf": 0.3,
        "title_boost": 0.5,
        "meta_boost": 0.1,
    }

    # Cross-encoder training
    TRAIN_CROSS_ENCODER = True
    HARD_K = 20
    NEG_PER_QUERY = 4
    RANDOM_NEG_PER_QUERY = 2
    CE_EPOCHS = 3

    SEED = 42


# ==========================================================
# ENV SETUP
# ==========================================================
def setup_environment(seed: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    logging.basicConfig(level=logging.ERROR)

    for lib in ["sentence_transformers", "transformers", "faiss"]:
        logging.getLogger(lib).setLevel(logging.ERROR)

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================================
# DATA
# ==========================================================
def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    sparse_docs = [d["clean_sparse"] for d in doc_json]
    dense_docs = [d["clean_dense"] for d in doc_json]
    title_tokenized = [set(d.get("title_tokens", [])) for d in doc_json]

    return doc_json, sparse_docs, dense_docs, title_tokenized


def load_train_data(path):
    qrels_json = load_qrels_json(path)
    qrels = build_qrels_mapping(qrels_json)
    queries = extract_queries(qrels_json)
    return queries, qrels, qrels_json


# ==========================================================
# BUILD COMPONENTS
# ==========================================================
def build_bm25(sparse_docs):
    tokenized = [turkish_preprocess(doc) for doc in sparse_docs]
    return BM25Okapi(tokenized)


def build_dense_index(dense_docs, model_name):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        dense_docs,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def build_cross_encoder(model_name):
    return CrossEncoder(model_name)


# ==========================================================
# BUILD IR MODEL
# ==========================================================
def build_ir_model(cfg: Config) -> UniversityIR:
    doc_json, sparse_docs, dense_docs, title_tokenized = load_corpus(cfg.CORPUS_PATH)

    bm25 = build_bm25(sparse_docs)
    dense_model, faiss_index = build_dense_index(dense_docs, cfg.DENSE_MODEL_NAME)
    cross_encoder = build_cross_encoder(cfg.CROSS_ENCODER_BASE)

    return UniversityIR(
        doc_json=doc_json,
        sparse_docs=sparse_docs,
        dense_docs=dense_docs,
        title_tokenized=title_tokenized,
        bm25_model=bm25,
        dense_model=dense_model,
        faiss_index=faiss_index,
        cross_encoder=cross_encoder,
        best_alpha=cfg.DEFAULT_ALPHA,
        alpha_by_type=cfg.DEFAULT_ALPHA_BY_TYPE.copy(),
        best_weights=cfg.DEFAULT_WEIGHTS.copy(),
    )


# ==========================================================
# TRAIN
# ==========================================================
def train_ir_model(
    ir_model: UniversityIR,
    cfg: Config
) -> Tuple[UniversityIR, Optional[CrossEncoder]]:

    queries, qrels, qrels_json = load_train_data(cfg.QREL_TRAIN_PATH)

    # --------------------------
    # Tune retrieval
    # --------------------------
    ir_model.best_alpha = tune_alpha(ir_model, queries, qrels)
    ir_model.alpha_by_type = learn_alpha_by_type(ir_model, queries, qrels)

    best_weights = tune_feature_weights(ir_model, queries, qrels)
    ir_model.best_weights.update(best_weights)

    # --------------------------
    # Cross-encoder training
    # --------------------------
    custom_ce = None

    if cfg.TRAIN_CROSS_ENCODER:
        setup_environment(cfg.SEED)
        custom_ce = train_cross_encoder_with_hard_negatives(
            qrels_json=qrels_json,
            doc_json=ir_model.doc_json,
            ir_model=ir_model,
            hard_k=cfg.HARD_K,
            neg_per_query=cfg.NEG_PER_QUERY,
            random_neg_per_query=cfg.RANDOM_NEG_PER_QUERY,
            epochs=cfg.CE_EPOCHS,
        )

        ir_model.cross_encoder_model = custom_ce

    return ir_model, custom_ce


# ==========================================================
# MAIN
# ==========================================================
def main():
    cfg = Config()
    setup_environment(cfg.SEED)

    # --------------------------
    # Build
    # --------------------------
    ir_model = build_ir_model(cfg)

    # --------------------------
    # Train
    # --------------------------
    ir_model, ce = train_ir_model(ir_model, cfg)

    # --------------------------
    # Save
    # --------------------------
    save_university_ir(ir_model, "saved_ir/ir_model")
    save_university_ir(ce, "saved_ir/ir_model_custom_ce")

    if ce:
        print("✅ Custom cross-encoder trained and attached")


if __name__ == "__main__":
    main()
