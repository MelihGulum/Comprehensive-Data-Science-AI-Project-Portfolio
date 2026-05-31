import numpy as np
import pandas as pd
import sys
import os
import json

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from src.evaluation.metrics import reciprocal_rank, recall_at_k, ndcg_at_k
from src.utils.qrels_utils import build_qrels_mapping, load_qrels_json

# -----------------------------
# SETTINGS
# -----------------------------
RUNS_DIR = Path("outputs/runs")
QREL_TEST = Path("data/processed/qrels/scored/qrel_test_scored_FINAL_ai.json")

OUTPUT_DIR = Path("outputs/analysis/analysis_significance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5

# -----------------------------
# LOAD RUN JSON
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
# PER-QUERY SCORES
# -----------------------------
def get_per_query_scores(retrieved_dict, qrels, k=5):
    rr_list, recall_list, ndcg_list = [], [], []

    common_queries = set(retrieved_dict) & set(qrels)

    for q in common_queries:
        retrieved = retrieved_dict[q]
        relevant = qrels[q]

        rr_list.append(reciprocal_rank(retrieved, relevant))
        recall_list.append(recall_at_k(retrieved, relevant, k))
        ndcg_list.append(ndcg_at_k(retrieved, relevant, k))

    return {
        "MRR": np.array(rr_list),
        "Recall": np.array(recall_list),
        "nDCG": np.array(ndcg_list)
    }

# -----------------------------
# STATS
# -----------------------------
def paired_wilcoxon(a, b):
    return wilcoxon(a, b).pvalue

def cohens_d(a, b):
    diff = a - b
    return np.mean(diff) / (np.std(diff) + 1e-10)

# -----------------------------
# SIGNIFICANCE
# -----------------------------
def evaluate_significance(top_name, baseline_names, qrels):
    top_run = load_run(top_name)
    top_scores = get_per_query_scores(top_run, qrels)

    results = []

    for name in baseline_names:
        base_run = load_run(name)
        base_scores = get_per_query_scores(base_run, qrels)

        p_mrr = paired_wilcoxon(top_scores["MRR"], base_scores["MRR"])
        p_ndcg = paired_wilcoxon(top_scores["nDCG"], base_scores["nDCG"])

        d_mrr = cohens_d(top_scores["MRR"], base_scores["MRR"])
        d_ndcg = cohens_d(top_scores["nDCG"], base_scores["nDCG"])

        results.append({
            "Model": name,
            "p_MRR": p_mrr,
            "p_nDCG": p_ndcg,
            "d_MRR": d_mrr,
            "d_nDCG": d_ndcg
        })

    df = pd.DataFrame(results)

    # multiple testing correction
    for metric in ["p_MRR", "p_nDCG"]:
        df[metric + "_corr"] = multipletests(df[metric], method="holm")[1]

    # significance flags
    df["sig_MRR"] = df["p_MRR_corr"] < 0.05
    df["sig_nDCG"] = df["p_nDCG_corr"] < 0.05

    return df.sort_values("p_MRR_corr")

# -----------------------------
# MODEL SETS
# -----------------------------
TOP_NO_CE = "Hybrid_Meta_Title_Length"
BASELINES_NO_CE = [
    "BM25",
    "Dense",
    "Hybrid",
    "Hybrid_Meta",
    "Hybrid_Title"
]

TOP_CE = "Full"
BASELINES_CE = [
    "Hybrid_Meta_Title_Cross",
    "Hybrid_Meta_Title_Expansion_Cross",
    "Hybrid",
    "BM25"
]

# -----------------------------
# RUN
# -----------------------------
qrels_json = load_qrels_json(QREL_TEST)
qrels_dict = build_qrels_mapping(qrels_json)

df_no_ce = evaluate_significance(TOP_NO_CE, BASELINES_NO_CE, qrels_dict)
df_ce = evaluate_significance(TOP_CE, BASELINES_CE, qrels_dict)

# -----------------------------
# SAVE RESULTS
# -----------------------------
df_no_ce.to_csv(OUTPUT_DIR / "significance_no_ce.csv", index=False)
df_ce.to_csv(OUTPUT_DIR / "significance_with_ce.csv", index=False)

# Combined
df_no_ce["setting"] = "no_ce"
df_ce["setting"] = "with_ce"
df_all = pd.concat([df_no_ce, df_ce], ignore_index=True)

df_all.to_csv(OUTPUT_DIR / "significance_all.csv", index=False)

# -----------------------------
# LATEX TABLE (paper)
# -----------------------------
latex_table = df_all[[
    "Model", "setting",
    "p_MRR_corr", "p_nDCG_corr",
    "d_MRR", "d_nDCG",
    "sig_MRR", "sig_nDCG"
]].round(4).to_latex(index=False)

with open(OUTPUT_DIR / "significance_table.tex", "w") as f:
    f.write(latex_table)

# -----------------------------
# PRINT
# -----------------------------
print("\n=== WITHOUT CE ===")
print(df_no_ce)

print("\n=== WITH CE ===")
print(df_ce)

print(f"\n💾 Saved to: {OUTPUT_DIR}")