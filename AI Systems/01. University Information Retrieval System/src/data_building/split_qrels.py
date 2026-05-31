import json
import random
from pathlib import Path
from typing import Union

QREL_DIR = Path("data/processed/qrels")
QREL_DIR.mkdir(parents=True, exist_ok=True)


def build_qrels_and_split(
        input_file: Union[str, Path],
        corpus_meta_file: Union[str, Path],
        train_file: Union[str, Path],
        test_file: Union[str, Path],
        train_ratio: float = 0.8,
        random_seed: int = 42
) -> None:
    """
    Convert gold query pool into structured train/test qrels split.

    Logic:
    - Split ONLY non-hard queries (80/20)
    - Add ALL hard queries to test set
    """

    # -----------------------------
    # Load corpus metadata
    # -----------------------------
    with open(corpus_meta_file, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    corpus_meta_dict = {
        item["doc_id"]: {
            "regulation_title": item.get("regulation_title", ""),
            "faculty": item.get("faculty", ""),
            "degree_level": item.get("degree_level", "")
        }
        for item in corpus_data
    }

    # -----------------------------
    # Load gold queries
    # -----------------------------
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    qrels = []

    for item in data:
        query_id = item["query_id"]
        query_text = item["query"]
        query_type = item.get("type", "default")
        candidate_docs = item.get("candidate_doc_ids", [])

        queries.append({
            "query_id": query_id,
            "query": query_text,
            "type": query_type
        })

        for doc_id in candidate_docs:
            meta = corpus_meta_dict.get(doc_id, {})
            qrels.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": None,
                "regulation_title": meta.get("regulation_title", ""),
                "faculty": meta.get("faculty", ""),
                "degree_level": meta.get("degree_level", "")
            })

    # -----------------------------
    # Separate hard vs non-hard
    # -----------------------------
    hard_queries = [q for q in queries if q["type"] == "hard"]
    non_hard_queries = [q for q in queries if q["type"] != "hard"]

    # -----------------------------
    # Shuffle ONLY non-hard
    # -----------------------------
    random.seed(random_seed)
    random.shuffle(non_hard_queries)

    split_index = int(len(non_hard_queries) * train_ratio)

    train_queries = non_hard_queries[:split_index]
    test_queries = non_hard_queries[split_index:]

    # -----------------------------
    # Add ALL hard queries to test
    # -----------------------------
    test_queries = test_queries + hard_queries

    # -----------------------------
    # Build ID sets
    # -----------------------------
    train_query_ids = {q["query_id"] for q in train_queries}
    test_query_ids = {q["query_id"] for q in test_queries}

    # -----------------------------
    # Build qrels
    # -----------------------------
    train_qrels = [q for q in qrels if q["query_id"] in train_query_ids]
    test_qrels = [q for q in qrels if q["query_id"] in test_query_ids]

    # -----------------------------
    # Save outputs
    # -----------------------------
    Path(train_file).write_text(
        json.dumps({
            "queries": train_queries,
            "qrels": train_qrels
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    Path(test_file).write_text(
        json.dumps({
            "queries": test_queries,
            "qrels": test_qrels
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # -----------------------------
    # Debug prints
    # -----------------------------
    train_hard = sum(q["type"] == "hard" for q in train_queries)
    test_hard = sum(q["type"] == "hard" for q in test_queries)

    print("✅ Split completed successfully!")
    print(f"📁 Train file: {train_file}")
    print(f"   → Queries: {len(train_queries)} | Hard: {train_hard}")
    print(f"📁 Test file: {test_file}")
    print(f"   → Queries: {len(test_queries)} | Hard: {test_hard}")