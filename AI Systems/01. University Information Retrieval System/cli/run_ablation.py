# experiments/ablation/run_ablation.py

from pathlib import Path
import logging
import os
import json
import pandas as pd

from experiments.ablation.config import CONFIGS
from experiments.ablation.runner import run_ablation
from src.evaluation.metrics import run_ablation_evaluation
from src.utils.qrels_utils import (
    load_qrels_json,
    build_qrels_mapping,
    extract_queries,
)
from src.utils.load_ir import load_university_ir


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("data/processed/qrels/scored/qrel_test_scored_FINAL_ai.json")
OUTPUT_DIR = Path("outputs/ablation_results")
RUNS_DIR = Path("outputs/runs")
IR_MODEL_PATH = Path("models/ir_model")
IR_MODEL_CUSTOM_PATH = Path("models/ir_model_custom_ce")


# -----------------------------
# Logging & Environment Setup
# -----------------------------
def setup_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    logging.basicConfig(level=logging.ERROR)

    for lib in [
        "sentence_transformers",
        "transformers",
        "httpx",
        "urllib3",
        "faiss",
    ]:
        logging.getLogger(lib).setLevel(logging.ERROR)

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()


# -----------------------------
# Split Qrels by Query Type
# -----------------------------
def split_qrels_by_type(qrels_json, relevance_threshold=1):
    easy_qrels = {}
    hard_qrels = {}

    # query_id -> type
    query_type_map = {
        str(q["query_id"]): q.get("type", "unknown")
        for q in qrels_json.get("queries", [])
    }

    for item in qrels_json.get("qrels", []):
        qid = str(item["query_id"])
        doc_id = str(item["doc_id"])
        relevance = item.get("relevance", 0)

        if relevance < relevance_threshold:
            continue

        qtype = query_type_map.get(qid, "unknown")

        if qtype == "easy":
            easy_qrels.setdefault(qid, {})[doc_id] = relevance
        elif qtype == "hard":
            hard_qrels.setdefault(qid, {})[doc_id] = relevance

    return easy_qrels, hard_qrels


# -----------------------------
# 🔥 FIX: Map query_text -> query_id
# -----------------------------
def remap_results_to_qid(results, qrels_json):
    text_to_qid = {
        q["query"]: str(q["query_id"])
        for q in qrels_json.get("queries", [])
    }

    remapped = {}

    for cfg_name, retrieved_dict in results.items():
        new_dict = {}

        for query_text, docs in retrieved_dict.items():
            qid = text_to_qid.get(query_text)

            if qid is None:
                continue  # safety

            new_dict[qid] = docs

        remapped[cfg_name] = new_dict

    return remapped


# -----------------------------
# Core Logic
# -----------------------------
def run(ir_model, output_name: str):
    qrels_json = load_qrels_json(DATA_PATH)

    # original (unchanged)
    qrels = build_qrels_mapping(qrels_json)
    queries = extract_queries(qrels_json)

    # run ablation
    results, latency = run_ablation(
        ir_model,
        queries,
        CONFIGS,
        top_k=5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # SAVE RUN FILES (raw)
    # -----------------------------
    for config_name, retrieved_dict in results.items():
        safe_name = config_name.replace("+", "_").replace("(", "").replace(")", "")
        out_path = RUNS_DIR / f"{output_name}__{safe_name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(retrieved_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved run files to: {RUNS_DIR}")

    # =============================
    # 🔥 FIX: ALIGN RESULTS WITH QRELS
    # =============================
    results_qid = remap_results_to_qid(results, qrels_json)

    # =============================
    # SPLIT QRELS
    # =============================
    easy_qrels, hard_qrels = split_qrels_by_type(qrels_json)

    print("Easy qrels:", len(easy_qrels))
    print("Hard qrels:", len(hard_qrels))

    # =============================
    # 🔍 DEBUG CHECK (IMPORTANT)
    # =============================
    sample_cfg = list(results_qid.keys())[0]
    print("\n[DEBUG]")
    print("Sample result keys:", list(results_qid[sample_cfg].keys())[:5])
    print("Sample qrels keys:", list(qrels.keys())[:5])

    # =============================
    # EVALUATION
    # =============================
    df_all = run_ablation_evaluation(
        results,
        latency,
        qrels,
        k=5,
    )
    df_all["split"] = "overall"

    df_easy = run_ablation_evaluation(
        results_qid,
        latency,
        easy_qrels,
        k=5,
    )
    df_easy["split"] = "easy"

    df_hard = run_ablation_evaluation(
        results_qid,
        latency,
        hard_qrels,
        k=5,
    )
    df_hard["split"] = "hard"

    df_final = pd.concat([df_all, df_easy, df_hard], ignore_index=True)

    output_file = OUTPUT_DIR / f"ablation_results_{output_name}_by_type.csv"
    df_final.to_csv(output_file, index=False)

    print("\n===== FINAL SPLIT RESULTS =====")
    print(df_final)

    print(f"\nSaved results to: {output_file}")


# -----------------------------
# Entry Point
# -----------------------------
def main():
    setup_environment()

    # Default model
    ir_default = load_university_ir(folder=str(IR_MODEL_PATH))
    run(ir_default, output_name="ir_model")

    # Custom CE model
    ir_custom = load_university_ir(folder=str(IR_MODEL_CUSTOM_PATH))
    run(ir_custom, output_name="ir_model_custom_ce")


if __name__ == "__main__":
    main()
