import re
import unicodedata
from typing import List
from .query_expansion_stopwords import TURKISH_STOPWORDS

try:
    from zemberek import TurkishMorphology
except ImportError:
    TurkishMorphology = None

morphology = TurkishMorphology.create_with_defaults() if TurkishMorphology else None


def is_abbreviation(token: str) -> bool:
    return (
        token.isupper() and len(token) <= 6
    ) or (
        token.lower() != token and len(token) <= 6
    )


def safe_lemmatize(token: str) -> str:
    """
    Apply lemmatization conservatively
    """
    if morphology is None:
        return token

    try:
        analyses = morphology.analyze(token)

        if not analyses:
            return token

        best = analyses[0]
        lemma = best.get_stem()

        # ❗ Avoid destructive shortening
        if len(lemma) < 3:
            return token

        # ❗ Avoid losing too much info (AKTS → akt)
        if len(token) - len(lemma) > 2:
            return token

        return lemma

    except Exception:
        return token


def turkish_preprocess(text: str, aggressive: bool = True) -> List[str]:
    """
    Q1-level Turkish preprocessing

    aggressive=False → explanation mode (safe)
    aggressive=True  → retrieval mode (strong normalization)
    """

    # Normalize
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    text = re.sub(r"[^\w\s\-çğıöşüÇĞİÖŞÜ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    processed = []

    for token in tokens:

        # Skip stopwords
        if token in TURKISH_STOPWORDS:
            continue

        # Preserve abbreviations
        if is_abbreviation(token):
            processed.append(token.lower())
            continue

        if aggressive and len(token) > 4:
            lemma = safe_lemmatize(token)
        else:
            lemma = token

        if len(lemma) > 2:
            processed.append(lemma)

    return processed
