# metrics.py
import numpy as np
import pandas as pd
from typing import List


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / max(len(relevant), 1)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_config(retrieved_dict: dict, qrels: dict, k: int = 10) -> dict:
    rr_list, recall_list, ndcg_list = [], [], []
    for query_text, retrieved in retrieved_dict.items():
        relevant = qrels.get(query_text, [])
        rr_list.append(reciprocal_rank(retrieved, relevant))
        recall_list.append(recall_at_k(retrieved, relevant, k))
        ndcg_list.append(ndcg_at_k(retrieved, relevant, k))
    return {"MRR@%d" % k: np.mean(rr_list),
            "Recall@%d" % k: np.mean(recall_list),
            "nDCG@%d" % k: np.mean(ndcg_list)}


def run_ablation_evaluation(ablation_results, latency_stats, qrels, k=10):
    records = []

    for cfg_name, retrieved_dict in ablation_results.items():
        metrics = evaluate_config(retrieved_dict, qrels, k=k)

        # Add latency metrics
        metrics["Avg Latency (s)"] = latency_stats[cfg_name]["avg_latency"]
        metrics["P95 Latency (s)"] = latency_stats[cfg_name]["p95_latency"]
        metrics["Total Time (s)"] = latency_stats[cfg_name]["total_time"]

        metrics["Config"] = cfg_name
        records.append(metrics)

    df = pd.DataFrame(records)

    df = df[
        ["Config",
         f"MRR@{k}",
         f"Recall@{k}",
         f"nDCG@{k}",
         "Avg Latency (s)",
         "P95 Latency (s)",
         "Total Time (s)"]
    ]

    return df.sort_values(by=f"MRR@{k}", ascending=False)
