# qrels_utils.py
import json
import re
from typing import List, Dict
from src.processing.preprocessing import turkish_preprocess


def load_qrels_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_qrels_mapping(qrels_json: dict) -> Dict[str, List[str]]:
    id_to_query = {q["query_id"]: q["query"] for q in qrels_json["queries"]}
    qrels_dict = {}

    for rel in qrels_json["qrels"]:
        relevance = rel.get("relevance")

        # 🔥 Fix: treat None as relevant
        if relevance is not None and relevance <= 0:
            continue

        query_text = id_to_query[rel["query_id"]]
        qrels_dict.setdefault(query_text, []).append(rel["doc_id"])

    return qrels_dict


def extract_queries(qrels_json: dict) -> List[str]:
    return [q["query"] for q in qrels_json["queries"]]


def classify_query(query: str) -> str:
    q = query.lower()
    if re.search(r"\b\d{4}\b", q):
        return "date_query"
    if re.search(r"\b(madde|fıkra|bent|article)\b", q):
        return "legal_query"
    tokens = turkish_preprocess(q)
    if len(tokens) <= 2: return "short_query"
    if any(t in ["nasıl","nedir","ne","kaç","hangi"] for t in tokens):
        return "informational"
    if any(t in ["başvuru","kayıt","mezuniyet","staj"] for t in tokens):
        return "process_query"
    return "default"