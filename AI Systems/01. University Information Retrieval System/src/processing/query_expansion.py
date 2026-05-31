# query_expansion.py
from typing import Dict
from .preprocessing import turkish_preprocess
from .query_expansion_stopwords import HINT_EXPANSION, SYNONYM_EXPANSION, CONCEPT_EXPANSION, HIERARCHICAL_EXPANSION


def expand_query_weighted(query: str,
                          max_hint=6, max_syn=6,
                          max_concept=5, max_hier=3) -> Dict[str, list]:
    query = query.lower()
    original_tokens = turkish_preprocess(query)
    expansion_tokens = []

    for key in HINT_EXPANSION:
        if key in query:
            expansion_tokens.extend(HINT_EXPANSION[key][:max_hint])
    for t in original_tokens:
        if t in SYNONYM_EXPANSION:
            expansion_tokens.extend(SYNONYM_EXPANSION[t][:max_syn])
    for key in CONCEPT_EXPANSION:
        if key in query:
            expansion_tokens.extend(CONCEPT_EXPANSION[key][:max_concept])
    for t in original_tokens:
        if t in HIERARCHICAL_EXPANSION:
            expansion_tokens.extend(HIERARCHICAL_EXPANSION[t][:max_hier])

    expansion_tokens = turkish_preprocess(" ".join(expansion_tokens))
    expansion_tokens = list(dict.fromkeys(expansion_tokens))
    return {"original": original_tokens, "expanded": expansion_tokens}
