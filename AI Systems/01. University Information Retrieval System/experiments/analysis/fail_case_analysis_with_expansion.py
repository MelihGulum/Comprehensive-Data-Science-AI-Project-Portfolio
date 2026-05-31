import json
import sys
import os
from collections import Counter, defaultdict

import pandas as pd

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.processing.query_expansion import expand_query_weighted

# ======================
# CONFIG
# ======================

BM25_PATH = "outputs/runs/ir_model__BM25.json"
DENSE_PATH = "outputs/runs/ir_model__Dense.json"

QRELS_PATH = "data/processed/qrels/qrel_test.json"
QUERIES_PATH = "data/processed/queries.json"
CORPUS_PATH = "data/processed/corpus_preprocessed.json"

TOP_K = 1

OUTPUT_DIR = "outputs/analysis/analysis_expansion"
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
        qrels_dict.setdefault(qid, set()).add(item["doc_id"])
    return qrels_dict

# ======================
# QUERY (EXPANSION)
# ======================

def normalize_query_with_expansion(text):
    expanded = expand_query_weighted(text)
    tokens = expanded["original"] + expanded["expanded"]
    return " ".join(tokens)

def build_queryid_to_text(queries_raw):
    return {
        str(item["query_id"]): normalize_query_with_expansion(item["query"])
        for item in queries_raw
    }

def remap_run_keys(run, query_map):
    text_to_qid = {v: k for k, v in query_map.items()}
    new_run = {}

    for query_text, docs in run.items():
        normalized = normalize_query_with_expansion(query_text)
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

def success_at_1(docs, relevant):
    return docs and docs[0] in relevant

# ======================
# FAILURE TAXONOMY
# ======================

def analyze_failure_reasons(query, bm25_docs, dense_docs, relevant):
    reasons = []

    bm25_top = bm25_docs[0] if bm25_docs else None
    dense_top = dense_docs[0] if dense_docs else None

    if bm25_top not in relevant and dense_top in relevant:
        reasons.append("keyword_mismatch")

    if dense_top not in relevant and bm25_top in relevant:
        reasons.append("semantic_gap")

    if bm25_top not in relevant and dense_top not in relevant:
        reasons.append("hard_query")

    if len(query.split()) > 6:
        reasons.append("long_query")

    return list(set(reasons))

# ======================
# MAIN
# ======================

def main():
    print("Loading data...")

    bm25 = load_json(BM25_PATH)
    dense = load_json(DENSE_PATH)

    qrels_raw = load_json(QRELS_PATH)
    if isinstance(qrels_raw, dict):
        qrels_raw = qrels_raw["qrels"]
    qrels = build_qrels_dict(qrels_raw)

    queries_raw = load_json(QUERIES_PATH)
    if isinstance(queries_raw, dict):
        queries_raw = queries_raw["queries"]

    query_map = build_queryid_to_text(queries_raw)

    bm25 = remap_run_keys(bm25, query_map)
    dense = remap_run_keys(dense, query_map)

    corpus_raw = load_json(CORPUS_PATH)
    corpus = build_corpus_dict(corpus_raw)

    stats = defaultdict(int)
    reason_counter = Counter()

    processed = 0
    rows = []

    for qid in qrels.keys():

        processed += 1

        query = query_map.get(qid, "")
        relevant = get_relevant(qrels, qid)

        bm25_docs = get_top_k(bm25, qid)
        dense_docs = get_top_k(dense, qid)

        bm25_1 = success_at_1(bm25_docs, relevant)
        dense_1 = success_at_1(dense_docs, relevant)

        if not bm25_1 and dense_1:
            stats["bm25_fail_dense_success"] += 1

        if bm25_1 and not dense_1:
            stats["dense_fail_bm25_success"] += 1

        if not bm25_1 and not dense_1:
            stats["both_fail"] += 1

        reasons = analyze_failure_reasons(query, bm25_docs, dense_docs, relevant)
        reason_counter.update(reasons)

        # SAVE QUERY LEVEL
        rows.append({
            "query_id": qid,
            "query": query,
            "bm25_top": bm25_docs[0] if bm25_docs else None,
            "dense_top": dense_docs[0] if dense_docs else None,
            "bm25_success@1": bm25_1,
            "dense_success@1": dense_1,
            "failure_types": ",".join(reasons)
        })

    # ======================
    # SAVE FILES
    # ======================

    df = pd.DataFrame(rows)
    df_path = os.path.join(OUTPUT_DIR, "query_level_expansion.csv")
    df.to_csv(df_path, index=False)

    stats_df = pd.DataFrame(stats.items(), columns=["case", "count"])
    stats_path = os.path.join(OUTPUT_DIR, "case_summary.csv")
    stats_df.to_csv(stats_path, index=False)

    failure_df = pd.DataFrame(reason_counter.items(), columns=["failure_type", "count"])
    failure_path = os.path.join(OUTPUT_DIR, "failure_distribution.csv")
    failure_df.to_csv(failure_path, index=False)

    print("\n💾 Saved:")
    print(" -", df_path)
    print(" -", stats_path)
    print(" -", failure_path)


if __name__ == "__main__":
    main()
