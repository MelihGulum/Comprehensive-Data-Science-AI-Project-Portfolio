import json
import sys
import os
from collections import defaultdict

import pandas as pd

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.processing.preprocessing import turkish_preprocess
from src.evaluation.metrics import reciprocal_rank, ndcg_at_k

# ======================
# CONFIG
# ======================

BM25_PATH = "outputs/runs/ir_model__BM25.json"
DENSE_PATH = "outputs/runs/ir_model__Dense.json"
HYBRID_PATH = "outputs/runs/ir_model__Hybrid.json"

QRELS_PATH = "data/processed/qrels/qrel_test.json"
QUERIES_PATH = "data/processed/queries.json"
CORPUS_PATH = "data/processed/corpus_preprocessed.json"

TOP_K = 5

OUTPUT_DIR = "outputs/analysis/analysis_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# UTILS
# ======================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================
# QRELS
# ======================

def build_qrels_dict(qrels_raw):
    qrels_dict = {}
    for item in qrels_raw:
        qid = str(item["query_id"])
        doc_id = str(item["doc_id"])
        qrels_dict.setdefault(qid, set()).add(doc_id)
    return qrels_dict

# ======================
# QUERY
# ======================

def normalize_query(text):
    return " ".join(turkish_preprocess(text))

def build_query_map(queries_raw):
    query_map = {}
    query_type_map = {}

    for item in queries_raw:
        qid = str(item["query_id"])
        query_map[qid] = normalize_query(item["query"])
        query_type_map[qid] = item.get("type", "unknown")

    return query_map, query_type_map

def remap_run_keys(run, query_map):
    text_to_qid = {v: k for k, v in query_map.items()}
    new_run = {}

    for query_text, docs in run.items():
        normalized = normalize_query(query_text)
        if normalized in text_to_qid:
            new_run[text_to_qid[normalized]] = docs

    return new_run

# ======================
# HELPERS
# ======================

def get_top_k(run, qid):
    docs = run.get(str(qid), [])
    if isinstance(docs, dict):
        docs = list(docs.keys())
    return [str(d) for d in docs[:TOP_K]]

def get_relevant(qrels, qid):
    return qrels.get(str(qid), set())

# ======================
# STAGE ANALYSIS
# ======================

def analyze_stage_failure(docs, relevant):
    retrieved = any(d in relevant for d in docs)
    top_correct = docs and docs[0] in relevant

    if not retrieved:
        return "retrieval_failure"
    elif not top_correct:
        return "ranking_failure"
    else:
        return "success"

# ======================
# SEVERITY
# ======================

def compute_severity(ndcg):
    if ndcg == 0:
        return "critical"
    elif ndcg < 0.3:
        return "high"
    elif ndcg < 0.7:
        return "medium"
    return "low"

# ======================
# MAIN
# ======================

def main():
    print("Loading data...\n")

    bm25 = load_json(BM25_PATH)
    dense = load_json(DENSE_PATH)
    hybrid = load_json(HYBRID_PATH)

    qrels_raw = load_json(QRELS_PATH)
    if isinstance(qrels_raw, dict):
        qrels_raw = qrels_raw["qrels"]
    qrels = build_qrels_dict(qrels_raw)

    queries_raw = load_json(QUERIES_PATH)
    if isinstance(queries_raw, dict):
        queries_raw = queries_raw["queries"]

    query_map, query_type_map = build_query_map(queries_raw)

    bm25 = remap_run_keys(bm25, query_map)
    dense = remap_run_keys(dense, query_map)
    hybrid = remap_run_keys(hybrid, query_map)

    stats = defaultdict(int)
    rows = []

    print("Running analysis...\n")

    for qid in qrels.keys():

        query = query_map.get(qid, "")
        query_type = query_type_map.get(qid, "unknown")
        relevant = get_relevant(qrels, qid)

        bm25_docs = get_top_k(bm25, qid)
        dense_docs = get_top_k(dense, qid)
        hybrid_docs = get_top_k(hybrid, qid)

        rr_bm25 = reciprocal_rank(bm25_docs, relevant)
        rr_dense = reciprocal_rank(dense_docs, relevant)
        rr_hybrid = reciprocal_rank(hybrid_docs, relevant)

        ndcg_bm25 = ndcg_at_k(bm25_docs, relevant, TOP_K)
        ndcg_dense = ndcg_at_k(dense_docs, relevant, TOP_K)
        ndcg_hybrid = ndcg_at_k(hybrid_docs, relevant, TOP_K)

        bm25_stage = analyze_stage_failure(bm25_docs, relevant)
        dense_stage = analyze_stage_failure(dense_docs, relevant)
        hybrid_stage = analyze_stage_failure(hybrid_docs, relevant)

        stats[f"bm25_{bm25_stage}"] += 1
        stats[f"dense_{dense_stage}"] += 1
        stats[f"hybrid_{hybrid_stage}"] += 1

        severity = compute_severity(ndcg_hybrid)
        difficulty = 1 - max(rr_bm25, rr_dense, rr_hybrid)

        best_model = max(
            [("bm25", rr_bm25), ("dense", rr_dense), ("hybrid", rr_hybrid)],
            key=lambda x: x[1]
        )[0]

        rows.append({
            "query_id": qid,
            "query": query,
            "query_type": query_type,

            "bm25_rr": rr_bm25,
            "dense_rr": rr_dense,
            "hybrid_rr": rr_hybrid,

            "bm25_ndcg": ndcg_bm25,
            "dense_ndcg": ndcg_dense,
            "hybrid_ndcg": ndcg_hybrid,

            "bm25_stage": bm25_stage,
            "dense_stage": dense_stage,
            "hybrid_stage": hybrid_stage,

            "severity": severity,
            "difficulty": difficulty,
            "best_model": best_model,

            "dense_vs_bm25_ndcg_diff": ndcg_dense - ndcg_bm25,
            "hybrid_vs_dense_ndcg_diff": ndcg_hybrid - ndcg_dense,
        })

    df = pd.DataFrame(rows)

    # ======================
    # SAVE CORE DATA
    # ======================

    main_path = os.path.join(OUTPUT_DIR, "stage_analysis.csv")
    df.to_csv(main_path, index=False)

    # ======================
    # SAVE AGGREGATES
    # ======================

    # Stage distribution
    stage_df = pd.DataFrame(stats.items(), columns=["stage", "count"])
    stage_path = os.path.join(OUTPUT_DIR, "stage_distribution.csv")
    stage_df.to_csv(stage_path, index=False)

    # Model performance summary
    summary_df = df.groupby("query_type")[[
        "bm25_ndcg", "dense_ndcg", "hybrid_ndcg",
        "bm25_rr", "dense_rr", "hybrid_rr"
    ]].mean()

    summary_path = os.path.join(OUTPUT_DIR, "performance_by_query_type.csv")
    summary_df.to_csv(summary_path)

    # Best model distribution
    best_model_df = df["best_model"].value_counts().reset_index()
    best_model_df.columns = ["model", "count"]

    best_model_path = os.path.join(OUTPUT_DIR, "best_model_distribution.csv")
    best_model_df.to_csv(best_model_path, index=False)

    # Severity distribution
    severity_df = df["severity"].value_counts().reset_index()
    severity_df.columns = ["severity", "count"]

    severity_path = os.path.join(OUTPUT_DIR, "severity_distribution.csv")
    severity_df.to_csv(severity_path, index=False)

    print("\n💾 Saved:")
    print(" -", main_path)
    print(" -", stage_path)
    print(" -", summary_path)
    print(" -", best_model_path)
    print(" -", severity_path)


if __name__ == "__main__":
    main()