# experiments/ablation/runner.py

import time
import numpy as np


def run_ablation(ir_model, queries, configs, top_k=10):
    results = {}
    latency_stats = {}

    for name, cfg in configs.items():
        print(f"[RUN] {name}")

        cfg_results = {}
        query_times = []

        start_total = time.time()

        for q in queries:
            start = time.time()

            retrieved = ir_model.retrieve_ablation(
                query=q,
                top_k=top_k,
                use_expansion=cfg.get("expansion", False),
                expansion_weight=cfg.get("expansion_weight"),
                use_meta=cfg.get("meta", False),
                use_title=cfg.get("title", False),
                use_length=cfg.get("length", False),
                use_cross=cfg.get("cross", False),
                use_dense=cfg.get("use_dense", True),
                alpha=cfg.get("alpha")
            )

            cfg_results[q] = retrieved
            query_times.append(time.time() - start)

        total_time = time.time() - start_total

        results[name] = cfg_results
        latency_stats[name] = {
            "avg_latency": np.mean(query_times),
            "p95_latency": np.percentile(query_times, 95),
            "total_time": total_time
        }

    return results, latency_stats
