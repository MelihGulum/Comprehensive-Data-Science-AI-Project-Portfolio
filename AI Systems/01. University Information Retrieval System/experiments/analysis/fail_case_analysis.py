import json
import sys
import os
from collections import Counter, defaultdict

import pandas as pd

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.processing.preprocessing import turkish_preprocess

# ======================
# CONFIG
# ======================

BM25_PATH = "outputs/runs/ir_model__BM25.json"
DENSE_PATH = "outputs/runs/ir_model__Dense.json"
HYBRID_PATH = "outputs/runs/ir_model__Hybrid.json"

HYBRID_WEIGHTED_EXPANSION_ZeroFive_CE_PATH = "outputs/runs/ir_model__Hybrid_WeightedExpansion0.5.json"
HYBRID_META_TITLE_EXPANSION_PATH = "outputs/runs/ir_model__Hybrid_Meta_Title_Expansion.json"
HYBRID_WEIGHTED_EXPANSION_ZeroThree_CE_PATH = "outputs/runs/ir_model__Hybrid_WeightedExpansion0.3.json"

CUSTOM_HYBRID_META_TITLE_CE_PATH = "outputs/runs/ir_model_custom_ce__Hybrid_Meta_Title_Cross.json"
CUSTOM_HYBRID_META_TITLE_EXPANSION_CE_PATH = "outputs/runs/ir_model_custom_ce__Hybrid_Meta_Title_Expansion_Cross.json"
CUSTOM_HYBRID_WEIGHTED_EXPANSION_PATH = "outputs/runs/ir_model_custom_ce__Hybrid_WeightedExpansion0.5.json"

RUN_PATHS = {
    "BM25": BM25_PATH,
    "Dense": DENSE_PATH,

    "Hybrid": HYBRID_PATH,
    "Hybrid + WeightedExpansion(0.5)": HYBRID_WEIGHTED_EXPANSION_ZeroFive_CE_PATH,
    "Hybrid + Expansion + CE": HYBRID_META_TITLE_EXPANSION_PATH,
    "Hybrid + WeightedExpansion(0.3)": HYBRID_WEIGHTED_EXPANSION_ZeroThree_CE_PATH,

    "Custom Hybrid + Meta + Title + CE": CUSTOM_HYBRID_META_TITLE_CE_PATH,
    "Custom Hybrid + Meta + Title + Expansion + CE": CUSTOM_HYBRID_META_TITLE_EXPANSION_CE_PATH,
    "Custom Hybrid + WeightedExpansion(0.5)": CUSTOM_HYBRID_WEIGHTED_EXPANSION_PATH,
}

QRELS_PATH = "data/processed/qrels/qrel_test.json"
QUERIES_PATH = "data/processed/queries.json"
CORPUS_PATH = "data/processed/corpus_preprocessed.json"

TOP_K = 5

OUTPUT_DIR = "outputs/analysis/fail_case_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# UTILS
# ======================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================
# BUILDERS
# ======================

def build_qrels_dict(qrels_raw):
    qrels_dict = {}
    for item in qrels_raw:
        qid = str(item["query_id"])
        qrels_dict.setdefault(qid, set()).add(item["doc_id"])
    return qrels_dict

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

def success_at_k(docs, relevant):
    return any(d in relevant for d in docs)

def success_at_1(docs, relevant):
    return docs and docs[0] in relevant

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

    runs = {name: load_json(path) for name, path in RUN_PATHS.items()}

    qrels_raw = load_json(QRELS_PATH)
    if isinstance(qrels_raw, dict):
        qrels_raw = qrels_raw["qrels"]
    qrels = build_qrels_dict(qrels_raw)

    queries_raw = load_json(QUERIES_PATH)
    if isinstance(queries_raw, dict):
        queries_raw = queries_raw["queries"]

    query_map = build_query_map(queries_raw)

    runs = {
        name: remap_run_keys(run, query_map)
        for name, run in runs.items()
    }

    corpus_raw = load_json(CORPUS_PATH)
    corpus = build_corpus_dict(corpus_raw) if isinstance(corpus_raw, list) else corpus_raw

    stats = defaultdict(int)
    reason_counter = Counter()
    processed = 0

    debug_rows = []

    for qid in qrels.keys():

        processed += 1

        query = query_map.get(qid, "")
        relevant = get_relevant(qrels, qid)

        model_docs = {
            name: get_top_k(run, qid)
            for name, run in runs.items()
        }

        model_success_1 = {
            name: success_at_1(docs, relevant)
            for name, docs in model_docs.items()
        }

        bm25_docs = model_docs.get("BM25", [])
        dense_docs = model_docs.get("Dense", [])
        hybrid_docs = model_docs.get("Hybrid", [])

        reasons = analyze_failure_reasons(
            query, bm25_docs, dense_docs, hybrid_docs, relevant
        )
        reason_counter.update(reasons)

        # SAVE DEBUG ROW
        debug_rows.append({
            "query_id": qid,
            "query": query,
            "reasons": ",".join(reasons),
            **{f"{name}_success@1": val for name, val in model_success_1.items()}
        })

    # ======================
    # SAVE FILES
    # ======================

    # Debug per query
    debug_df = pd.DataFrame(debug_rows)
    debug_path = os.path.join(OUTPUT_DIR, "query_level_analysis.csv")
    debug_df.to_csv(debug_path, index=False)

    # Failure distribution
    failure_df = pd.DataFrame(reason_counter.items(), columns=["failure_type", "count"])
    failure_path = os.path.join(OUTPUT_DIR, "failure_distribution.csv")
    failure_df.to_csv(failure_path, index=False)

    # Leaderboard
    leaderboard = []

    for name, run in runs.items():
        success = sum(
            success_at_1(get_top_k(run, qid), get_relevant(qrels, qid))
            for qid in qrels.keys()
        )
        leaderboard.append({
            "model": name,
            "success@1": success,
            "total": processed,
            "rate": success / processed
        })

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("rate", ascending=False)
    leaderboard_path = os.path.join(OUTPUT_DIR, "leaderboard.csv")
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\n💾 Saved:")
    print(" -", debug_path)
    print(" -", failure_path)
    print(" -", leaderboard_path)


if __name__ == "__main__":
    main()
