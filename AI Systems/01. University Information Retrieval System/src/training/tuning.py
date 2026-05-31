# tuning.py
from typing import List, Dict
import numpy as np
from .metrics import evaluate_config, reciprocal_rank
from .qrels_utils import classify_query


def tune_alpha(ir_model, queries: List[str], qrels: Dict[str, List[str]],
               alphas: List[float] = [0.3, 0.5, 0.7], k: int = 5) -> float:
    """Grid search to find best alpha for IR model using MRR@k."""
    best_alpha, best_score = None, -1
    for a in alphas:
        results = {q: ir_model.retrieve_ablation(query=q, top_k=k, alpha=a,
                                                 use_dense=True, use_meta=True,
                                                 use_title=True, use_length=True)
                   for q in queries}
        metrics = evaluate_config(results, qrels, k=k)
        if metrics[f"MRR@{k}"] > best_score:
            best_score = metrics[f"MRR@{k}"]
            best_alpha = a
    print("Best alpha:", best_alpha)
    return best_alpha


def tune_feature_weights(ir_model, queries: List[str], qrels: Dict[str, List[str]],
                         configs: List[Dict[str, float]] = None, k: int = 5) -> Dict[str, float]:
    """Search best feature weights (title/meta boosts)."""
    if configs is None:
        configs = [
            {"title": 0.3, "meta": 0.1},
            {"title": 0.5, "meta": 0.1},
            {"title": 0.5, "meta": 0.2},
            {"title": 0.7, "meta": 0.2},
        ]

    best_cfg, best_score = None, -1
    for cfg in configs:
        results = {q: ir_model.retrieve_ablation(query=q, top_k=k, alpha=0.5,
                                                 use_dense=True, use_meta=True,
                                                 use_title=True, use_length=True)
                   for q in queries}
        metrics = evaluate_config(results, qrels, k=k)
        if metrics[f"MRR@{k}"] > best_score:
            best_score = metrics[f"MRR@{k}"]
            best_cfg = cfg
    print("Best feature weights:", best_cfg)
    return best_cfg


def learn_alpha_by_type(ir_model, queries: List[str], qrels: Dict[str, List[str]],
                        alpha_candidates: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, float]:
    """Learn alpha per query type based on reciprocal rank."""
    type_alphas: Dict[str, List[float]] = {}
    for q in queries:
        qtype = classify_query(q)
        type_alphas.setdefault(qtype, [])
        best_local_alpha, best_score = None, -1
        for a in alpha_candidates:
            retrieved = ir_model.retrieve_ablation(q, alpha=a)
            score = reciprocal_rank(retrieved, qrels.get(q, []))
            if score > best_score:
                best_score = score
                best_local_alpha = a
        type_alphas[qtype].append(best_local_alpha)
    # Average alpha per query type
    type_alphas_avg = {t: float(np.mean(vals)) for t, vals in type_alphas.items()}
    print("Alpha per query type:", type_alphas_avg)
    return type_alphas_avg


def build_cross_encoder_data(qrels_json: dict, doc_json: List[dict]) -> List[tuple]:
    """Prepare (query, doc, label) data for cross-encoder training."""
    id_to_query = {q["query_id"]: q["query"] for q in qrels_json["queries"]}
    data = []
    for rel in qrels_json["qrels"]:
        q = id_to_query[rel["query_id"]]
        doc = next(d["text"] for d in doc_json if d["doc_id"] == rel["doc_id"])
        data.append((q, doc, rel["relevance"]))
    return data
