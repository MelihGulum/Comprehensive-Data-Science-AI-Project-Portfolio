# file: load_ir_safe.py
import json
import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.retrieval_pipeline import UniversityIR
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

CORPUS_PATH = os.path.join(ROOT_DIR := os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
                           "data", "processed", "corpus_preprocessed.json")


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)

    sparse_docs = [d["clean_sparse"] for d in doc_json]
    dense_docs = [d["clean_dense"] for d in doc_json]
    title_tokenized = [set(d.get("title_tokens", [])) for d in doc_json]

    return doc_json, sparse_docs, dense_docs, title_tokenized


def safe_load_json(file_path, expected_type):
    """Load JSON safely and validate type."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {expected_type} in {file_path}, got {type(data)}")
    return data


def safe_load_pickle(file_path, expected_type):
    """Load pickle safely with type validation."""
    import pickle
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {expected_type} in {file_path}, got {type(data)}")
    return data


def load_university_ir(folder="saved_ir", es_client=None, es_index=None):
    # --- Load JSON/serialized data safely ---

    doc_json, sparse_docs, dense_docs, title_tokenized = load_corpus(CORPUS_PATH)

    doc_ids = safe_load_pickle(f"{folder}/doc_ids.pkl", list)
    bm25_model = safe_load_pickle(f"{folder}/bm25.pkl", object)
    weights = safe_load_pickle(f"{folder}/weights.pkl", dict) or {}

    # --- Load dense model safely ---
    dense_model = None
    dense_model_path = os.path.join(folder, "dense_model")
    if os.path.exists(dense_model_path):
        try:
            dense_model = SentenceTransformer(dense_model_path)
        except Exception as e:
            print(f"Warning: failed to load dense model: {e}")

    # --- Load FAISS index safely ---
    faiss_index = None
    faiss_path = os.path.join(folder, "faiss.index")
    if os.path.exists(faiss_path):
        try:
            faiss_index = faiss.read_index(faiss_path)
        except Exception as e:
            print(f"Warning: failed to load FAISS index: {e}")

    # --- Load cross-encoder safely ---
    cross_encoder = None
    ce_path = os.path.join(folder, "cross_encoder")
    if os.path.exists(ce_path):
        try:
            cross_encoder = CrossEncoder(ce_path)
        except Exception as e:
            print(f"Warning: failed to load cross-encoder: {e}")

    # --- Reconstruct UniversityIR ---
    ir = UniversityIR(
        doc_json=doc_json,
        sparse_docs=sparse_docs,
        dense_docs=dense_docs,
        title_tokenized=title_tokenized,
        bm25_model=bm25_model,
        dense_model=dense_model,
        faiss_index=faiss_index,
        cross_encoder=cross_encoder,
        es_client=es_client,
        es_index=es_index,
        best_alpha=weights.get("best_alpha", 0.5),
        alpha_by_type=weights.get("alpha_by_type"),
        best_weights=weights.get("best_weights")
    )

    # --- Inject FAISS-aligned doc_ids ---
    ir.doc_ids = doc_ids
    ir.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    print("IR model loaded safely.")
    return ir
