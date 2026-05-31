from typing import List, Tuple, Dict, Any
import random
from collections import defaultdict

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


# ==========================================================
# DATA BUILDER
# ==========================================================
def build_cross_encoder_data_hard_negatives(
    qrels_json: Dict[str, Any],
    doc_json: List[Dict[str, Any]],
    ir_model,
    hard_k: int = 15,
    neg_per_query: int = 3,
    random_neg_per_query: int = 2,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """
    Build training data with:
    - positives from qrels
    - hard negatives from BM25
    - optional random negatives

    Returns:
        List of (query, document_text, label)
    """

    random.seed(seed)

    # -------------------------
    # Build doc lookup
    # -------------------------
    doc_map = {d["doc_id"]: d["text"] for d in doc_json}
    all_doc_ids = list(doc_map.keys())

    training_data: List[Tuple[str, str, float]] = []

    for query, doc_list in qrels_json.items():

        # -------------------------
        # POSITIVES
        # -------------------------
        positives = {
            item["doc_id"]
            for item in doc_list
            if item.get("relevance", 0) > 0
        }

        for doc_id in positives:
            if doc_id in doc_map:
                training_data.append((query, doc_map[doc_id], 1.0))

        # -------------------------
        # HARD NEGATIVES (BM25)
        # -------------------------
        bm25_candidates = ir_model.bm25_only(query, top_k=hard_k)

        hard_pool = [
            doc_id for doc_id in bm25_candidates
            if doc_id not in positives and doc_id in doc_map
        ]

        if hard_pool:
            sampled_hard = random.sample(
                hard_pool,
                min(neg_per_query, len(hard_pool))
            )

            for doc_id in sampled_hard:
                training_data.append((query, doc_map[doc_id], 0.0))

        # -------------------------
        # RANDOM NEGATIVES
        # -------------------------
        if random_neg_per_query > 0:
            random_pool = [
                doc_id for doc_id in all_doc_ids
                if doc_id not in positives
            ]

            if random_pool:
                sampled_random = random.sample(
                    random_pool,
                    min(random_neg_per_query, len(random_pool))
                )

                for doc_id in sampled_random:
                    training_data.append((query, doc_map[doc_id], 0.0))

    return training_data


# ==========================================================
# DATASET UTILS
# ==========================================================
def to_input_examples(
    train_data: List[Tuple[str, str, float]]
) -> List[InputExample]:
    return [
        InputExample(texts=[q, d], label=float(label))
        for q, d, label in train_data
    ]


def build_dataloader(
    samples: List[InputExample],
    batch_size: int = 16,
) -> DataLoader:
    return DataLoader(samples, shuffle=True, batch_size=batch_size)


# ==========================================================
# TRAINING
# ==========================================================
def train_cross_encoder(
    train_data: List[Tuple[str, str, float]],
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  #"cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 100,
    show_progress_bar: bool = True,
) -> CrossEncoder:
    """
    Train a domain-adapted cross-encoder.

    Fully configurable & reusable.
    """

    model = CrossEncoder(model_name)

    samples = to_input_examples(train_data)
    loader = build_dataloader(samples, batch_size=batch_size)

    model.fit(
        train_dataloader=loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=show_progress_bar,
    )

    return model


# ==========================================================
# HIGH-LEVEL PIPELINE (OPTIONAL HELPER)
# ==========================================================
def train_cross_encoder_with_hard_negatives(
    qrels_json,
    doc_json,
    ir_model,
    hard_k: int = 15,
    neg_per_query: int = 3,
    random_neg_per_query: int = 2,
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  #"cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
) -> CrossEncoder:
    """
    End-to-end helper:
    build data → train model
    """

    train_data = build_cross_encoder_data_hard_negatives(
        qrels_json=qrels_json,
        doc_json=doc_json,
        ir_model=ir_model,
        hard_k=hard_k,
        neg_per_query=neg_per_query,
        random_neg_per_query=random_neg_per_query,
    )

    model = train_cross_encoder(
        train_data=train_data,
        model_name=model_name,
        epochs=epochs,
    )

    return model
