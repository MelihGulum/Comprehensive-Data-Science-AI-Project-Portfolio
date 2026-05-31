from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
import os
import re

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.evaluation.metrics import reciprocal_rank, recall_at_k, ndcg_at_k
from src.processing.preprocessing import turkish_preprocess
from src.utils.qrels_utils import load_qrels_json, build_qrels_mapping

# -----------------------------
# SETTINGS
# -----------------------------
RUNS_DIR = Path("outputs/runs")
QREL_TEST = Path("data/processed/qrels/qrel_test.json")
K = 5

OUTPUT_DIR = Path("outputs/analysis/analysis_query_type")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD QRELS
# -----------------------------
qrels_json = load_qrels_json(QREL_TEST)
qrels_dict = build_qrels_mapping(qrels_json)

# -----------------------------
# LOAD RUN FILES
# -----------------------------
def load_run(name: str, prefix="ir_model") -> dict:
    import glob
    file_pattern = f"{RUNS_DIR}/{prefix}__{name}.json"
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No run file found for {name}")

    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# LOAD ALL MODELS
# -----------------------------
runs = {
    "BM25": load_run("BM25"),
    "Dense": load_run("Dense"),
    "Hybrid": load_run("Hybrid_Meta_Title_Length"),
    "CE_Full": load_run("Full", prefix="ir_model_custom_ce")
}

# -----------------------------
# QUERY TYPE CLASSIFIER
# -----------------------------
def classify_query(query):
    q = query.lower()

    if re.search(r"\b\d{4}\b", q):
        return "date_query"

    if re.search(r"\b(madde|fıkra|bent|article)\b", q):
        return "legal_query"

    tokens = turkish_preprocess(q)

    if len(tokens) <= 2:
        return "short_query"

    if any(t in ["nasıl", "nedir", "ne", "kaç", "hangi"] for t in tokens):
        return "informational"

    if any(t in ["başvuru", "kayıt", "mezuniyet", "staj"] for t in tokens):
        return "process_query"

    return "default"

# -----------------------------
# EVALUATE BY TYPE
# -----------------------------
def evaluate_by_query_type(run_dict, qrels, k=5):
    data = {}

    for q, retrieved in run_dict.items():
        qtype = classify_query(q)
        rel = qrels.get(q, [])

        if qtype not in data:
            data[qtype] = {"MRR": [], "nDCG": [], "Recall": []}

        data[qtype]["MRR"].append(reciprocal_rank(retrieved, rel))
        data[qtype]["nDCG"].append(ndcg_at_k(retrieved, rel, k))
        data[qtype]["Recall"].append(recall_at_k(retrieved, rel, k))

    rows = []
    for t, vals in data.items():
        rows.append({
            "Type": t,
            "MRR": np.mean(vals["MRR"]),
            "nDCG": np.mean(vals["nDCG"]),
            "Recall": np.mean(vals["Recall"])
        })

    return pd.DataFrame(rows)

# -----------------------------
# RUN ALL MODELS
# -----------------------------
results = {}

for name, run in runs.items():
    df = evaluate_by_query_type(run, qrels_dict, k=K)
    df["Model"] = name
    results[name] = df

df_all = pd.concat(results.values(), ignore_index=True)

# -----------------------------
# PIVOTS
# -----------------------------
pivot_mrr = df_all.pivot(index="Type", columns="Model", values="MRR")
pivot_ndcg = df_all.pivot(index="Type", columns="Model", values="nDCG")
pivot_recall = df_all.pivot(index="Type", columns="Model", values="Recall")

# -----------------------------
# SAVE EVERYTHING
# -----------------------------

# Raw long format
df_all.to_csv(OUTPUT_DIR / "query_type_full.csv", index=False)

# Pivot tables
pivot_mrr.to_csv(OUTPUT_DIR / "pivot_mrr.csv")
pivot_ndcg.to_csv(OUTPUT_DIR / "pivot_ndcg.csv")
pivot_recall.to_csv(OUTPUT_DIR / "pivot_recall.csv")

# Combined table (çok kullanışlı)
combined = pd.concat({
    "MRR": pivot_mrr,
    "nDCG": pivot_ndcg,
    "Recall": pivot_recall
}, axis=1)

combined.to_csv(OUTPUT_DIR / "combined_metrics.csv")

# -----------------------------
# 🔥 LATEX TABLE (paper ready)
# -----------------------------
latex_table = pivot_ndcg.round(3).to_latex()

with open(OUTPUT_DIR / "table_ndcg.tex", "w") as f:
    f.write(latex_table)

# -----------------------------
# PRINT
# -----------------------------
print("\n=== MRR by Query Type ===")
print(pivot_mrr.round(3))

print("\n=== nDCG@5 by Query Type ===")
print(pivot_ndcg.round(3))

print("\n=== Recall@5 by Query Type ===")
print(pivot_recall.round(3))

print(f"\n💾 Saved to: {OUTPUT_DIR}")