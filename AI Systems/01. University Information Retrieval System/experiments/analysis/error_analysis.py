import json
import sys
import os
from collections import Counter, defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

OUTPUT_DIR = "outputs/analysis/error_analysis"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

sns.set(style="whitegrid")

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
        qrels_dict.setdefault(qid, set()).add(item["doc_id"])
    return qrels_dict

# ======================
# QUERY
# ======================

def normalize_query(text):
    return " ".join(turkish_preprocess(text))

def build_query_map(queries_raw):
    return {
        str(item["query_id"]): normalize_query(item["query"])
        for item in queries_raw
    }

def remap_run_keys(run, query_map):
    text_to_qid = {v: k for k, v in query_map.items()}
    new_run = {}

    for query_text, docs in run.items():
        normalized = normalize_query(query_text)
        if normalized in text_to_qid:
            new_run[text_to_qid[normalized]] = docs

    return new_run

# ======================
# CORPUS
# ======================

def build_corpus_dict(corpus_raw):
    return {
        item["doc_id"]: item.get("text") or item.get("content") or ""
        for item in corpus_raw
    }

# ======================
# HELPERS
# ======================

def get_top_k(run, qid):
    docs = run.get(str(qid), [])
    if isinstance(docs, dict):
        return list(docs.keys())[:TOP_K]
    return docs[:TOP_K]

def get_relevant(qrels, qid):
    return qrels.get(str(qid), set())

# ======================
# FAILURE TAXONOMY
# ======================

def analyze_failure_reasons(query, bm25_docs, dense_docs, hybrid_docs, relevant):
    reasons = []

    bm25_top = bm25_docs[0] if bm25_docs else None
    dense_top = dense_docs[0] if dense_docs else None
    hybrid_top = hybrid_docs[0] if hybrid_docs else None

    if bm25_top not in relevant and dense_top in relevant:
        reasons.append("keyword_mismatch")

    if dense_top not in relevant and bm25_top in relevant:
        reasons.append("semantic_gap")

    if bm25_top not in relevant and dense_top not in relevant:
        reasons.append("hard_query")

    if any(d in relevant for d in hybrid_docs) and hybrid_top not in relevant:
        reasons.append("ranking_error")

    if len(query.split()) > 6:
        reasons.append("long_query")

    return list(set(reasons))

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

    query_map = build_query_map(queries_raw)

    bm25 = remap_run_keys(bm25, query_map)
    dense = remap_run_keys(dense, query_map)
    hybrid = remap_run_keys(hybrid, query_map)

    corpus_raw = load_json(CORPUS_PATH)
    corpus = build_corpus_dict(corpus_raw) if isinstance(corpus_raw, list) else corpus_raw

    analysis_rows = []
    reason_counter = Counter()

    print("Running merged analysis...\n")

    for qid in qrels.keys():

        query = query_map.get(qid, "")
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

        reasons = analyze_failure_reasons(
            query, bm25_docs, dense_docs, hybrid_docs, relevant
        )
        reason_counter.update(reasons)

        analysis_rows.append({
            "query_id": qid,
            "query": query,

            "bm25_rr": rr_bm25,
            "dense_rr": rr_dense,
            "hybrid_rr": rr_hybrid,

            "bm25_ndcg": ndcg_bm25,
            "dense_ndcg": ndcg_dense,
            "hybrid_ndcg": ndcg_hybrid,

            "dense_vs_bm25_rr_diff": rr_dense - rr_bm25,
            "dense_vs_bm25_ndcg_diff": ndcg_dense - ndcg_bm25,

            "hybrid_vs_bm25_rr_diff": rr_hybrid - rr_bm25,
            "hybrid_vs_bm25_ndcg_diff": ndcg_hybrid - ndcg_bm25,

            "hybrid_vs_dense_rr_diff": rr_hybrid - rr_dense,
            "hybrid_vs_dense_ndcg_diff": ndcg_hybrid - ndcg_dense,

            "failure_types": ",".join(reasons)
        })

    df = pd.DataFrame(analysis_rows)

    # ======================
    # SAVE DATA
    # ======================

    df_path = os.path.join(OUTPUT_DIR, "ir_analysis.csv")
    df.to_csv(df_path, index=False)
    print(f"\n💾 Analysis saved to: {df_path}")

    failure_df = pd.DataFrame(reason_counter.items(), columns=["failure_type", "count"])
    failure_path = os.path.join(OUTPUT_DIR, "failure_distribution.csv")
    failure_df.to_csv(failure_path, index=False)
    print(f"💾 Failure distribution saved to: {failure_path}")

    df_expanded = df.copy()
    df_expanded["failure_types"] = df_expanded["failure_types"].str.split(",")
    df_expanded = df_expanded.explode("failure_types")

    impact_df = df_expanded.groupby("failure_types")[
        [
            "hybrid_vs_bm25_ndcg_diff",
            "dense_vs_bm25_ndcg_diff",
            "hybrid_vs_dense_ndcg_diff"
        ]
    ].mean()

    impact_path = os.path.join(OUTPUT_DIR, "failure_impact.csv")
    impact_df.to_csv(impact_path)
    print(f"💾 Failure impact saved to: {impact_path}")

    # ======================
    # PLOT
    # ======================

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="dense_vs_bm25_rr_diff",
        y="dense_vs_bm25_ndcg_diff",
        label="Dense vs BM25"
    )

    sns.scatterplot(
        data=df,
        x="hybrid_vs_bm25_rr_diff",
        y="hybrid_vs_bm25_ndcg_diff",
        label="Hybrid vs BM25"
    )

    plt.axvline(0, linestyle="--")
    plt.axhline(0, linestyle="--")

    plt.title("Model Comparisons (Per-query differences)")
    plt.xlabel("MRR Difference")
    plt.ylabel("nDCG Difference")

    plt.legend()

    fig_path = os.path.join(FIGURE_DIR, "model_comparison_scatter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Figure saved to: {fig_path}")

    plt.close()


if __name__ == "__main__":
    main()
