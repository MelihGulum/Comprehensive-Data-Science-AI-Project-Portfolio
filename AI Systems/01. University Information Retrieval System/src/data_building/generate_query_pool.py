"""
gold_query_pool.py

Generates a high-quality gold query pool for university regulations IR:

- Hybrid retrieval: BM25 + dense embeddings + cross-encoder reranking
- Weighted PRF and phrase-aware query expansion
- Turkish-specific preprocessing and stopword removal

Usage:
from gold_query_pool import GoldQueryPool

gqp = GoldQueryPool(
    corpus_path="data/processed/corpus_preprocessed.json",
    queries_path="data/processed/queries.json",
    max_pool_size=200
)
gqp.build_pool()
gqp.save("data/processed/gold_query_pool.json")
"""

import re
import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Turkish NLP
try:
    from zemberek import TurkishMorphology
except ImportError:
    TurkishMorphology = None
    logging.warning("⚠️ Zemberek not installed. Run `pip install zemberek-python` for stemming.")

# ======================
# STOPWORDS
# ======================
TURKISH_STOPWORDS = {
    "ve", "ile", "ama", "ancak", "fakat", "lakin", "veya", "ya", "yahut", "çünkü", "ise", "ki", "dahi",
    "için", "olarak", "bu", "şu", "o", "bunlar", "şunlar", "onlar", "her", "bazı", "tüm", "hiç",
    "kendi", "mi", "mı", "mu", "mü", "çok", "daha", "en", "az", "son", "önce", "sonra", "kadar",
    "gibi", "fıkra", "bent", "sayılı", "tarihli", "uyarınca", "gereğince", "kapsamında", "ilişkin",
    "dair", "hakkında", "yapılır", "edilir", "olur", "olduğu", "amacıyla", "şekilde", "hususunda",
    "etmek", "olmak", "yapmak", "bulunmak", "göstermek", "sunmak", "belirlemek"
}

morphology = TurkishMorphology.create_with_defaults() if TurkishMorphology else None


# ======================
# GOLD QUERY POOL CLASS
# ======================
class GoldQueryPool:
    def __init__(self, corpus_path, queries_path, max_pool_size=200):
        self.corpus_path = Path(corpus_path)
        self.queries_path = Path(queries_path)
        self.max_pool_size = max_pool_size

        # Data holders
        self.doc_json = []
        self.query_json = []
        self.sparse_docs = []
        self.dense_docs = []
        self.doc_ids = []
        self.queries = []
        self.query_ids = []
        self.query_types = []

        self.tokenized_corpus = []
        self.title_tokenized = []

        # Models / indices
        self.bm25 = None
        self.dense_model = None
        self.index = None
        self.cross_encoder = None

        # Load data on init
        self._load_data()
        self._preprocess_titles()
        self._init_models()

    # -----------------------------
    # DATA LOADING
    # -----------------------------
    def _load_data(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.doc_json = json.load(f)

        with open(self.queries_path, "r", encoding="utf-8") as f:
            self.query_json = json.load(f)

        self.sparse_docs = [d["clean_sparse"] for d in self.doc_json]
        self.dense_docs = [d["clean_dense"] for d in self.doc_json]
        self.doc_ids = [d["doc_id"] for d in self.doc_json]

        self.queries = [q["query"] for q in self.query_json]
        self.query_ids = [q["query_id"] for q in self.query_json]
        self.query_types = [q.get("type", "default") for q in self.query_json]

        print(
            f"Loaded {len(self.sparse_docs)} sparse docs, {len(self.dense_docs)} dense docs, {len(self.queries)} queries")

    # -----------------------------
    # TURKISH PREPROCESSING
    # -----------------------------
    @staticmethod
    def turkish_preprocess(text, keep_dates=True):
        text = unicodedata.normalize("NFKC", text).lower()
        text = ''.join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

        phrase_map = {
            r'genel\s+not\s+ortalamas[ıi]': 'gno',
            r'azami\s+ogrenim\s+sures[ıi]': 'azami_sure',
            r'yatay\s+gecis': 'yatay_gecis',
            r'dikey\s+gecis': 'dikey_gecis',
            r'kayit\s+silme': 'kayit_silme',
            r'tekirdag\s+namik\s+kemal\s+universites[ıi](nde|nin|ne|ni)?': 'tnku tekirdag_namik_kemal_universitesi',
            r'namik\s+kemal\s+universites[ıi](nde|nin|ne|ni)?': 'tnku tekirdag_namik_kemal_universitesi',
            r'\bnku\b': 'tnku tekirdag_namik_kemal_universitesi',
            r'\btnku\b': 'tnku tekirdag_namik_kemal_universitesi',
        }
        for pattern, replacement in phrase_map.items():
            text = re.sub(pattern, replacement, text)

        if keep_dates:
            text = re.sub(r"[^\w\s\-\d]", " ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        processed_tokens = []

        for token in tokens:
            if not token.strip() or token in TURKISH_STOPWORDS:
                continue

            # Morphological stemming
            if morphology:
                try:
                    analyses = morphology.analyze(token)
                    lemma = analyses[0].get_stem() if analyses else token
                except Exception:
                    lemma = token
            else:
                lemma = token

            if len(lemma) > 2:
                processed_tokens.append(lemma)
        return processed_tokens

    # -----------------------------
    # TITLE TOKENIZATION
    # -----------------------------
    def _preprocess_titles(self):
        self.title_tokenized = []
        for doc in self.doc_json:
            if "title_tokens" in doc:
                processed = [t for t in self.turkish_preprocess(" ".join(doc["title_tokens"]))]
            else:
                processed = []
            self.title_tokenized.append(set(processed))

    # -----------------------------
    # MODEL INITIALIZATION
    # -----------------------------
    def _init_models(self):
        # BM25
        self.tokenized_corpus = [self.turkish_preprocess(doc) for doc in self.sparse_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense + FAISS
        self.dense_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        doc_embeddings = self.dense_model.encode(self.dense_docs, convert_to_numpy=True, show_progress_bar=True)
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(doc_embeddings)
        self.index.add(doc_embeddings)

        # Cross-encoder
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # -----------------------------
    # QUERY EXPANSION
    # -----------------------------
    @staticmethod
    def unique_preserve_order(tokens):
        seen = set()
        result = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def expand_query(self, query, max_hint=6, max_syn=6, max_concept=5, max_hier=3):
        # Expansion dictionaries
        hint_expansion = {
            "ders kaydı": ["akademik takvim", "kayıt tarihleri", "ders ekleme çıkarma"],
            "bahar dönemi başlangıç": ["akademik takvim", "eğitim yılı takvimi"],
            "final sınav": ["yarıyıl sonu sınavı", "sınav takvimi", "sınav tarihleri"],
            "mezuniyet": ["diploma hakkı", "staj bitirme projesi", "mezuniyet koşulları"],
        }
        synonym_expansion = {
            "şart": ["koşul", "gereklilik"],
            "koşul": ["şart", "gereklilik"],
            "madde": ["hüküm"],
            "başvuru": ["müracaat", "form", "dilekçe"],
            "kayıt silme": ["ilişik kesme", "kayıt iptali", "çıkış"],
            "mezuniyet": ["diploma hakkı", "staj", "bitirme projesi"],
            "ders kaydı": ["kayıt", "ders seçimi", "ekle/çıkar"],
            "sınav": ["ara sınav", "final", "bütünleme", "sınav tarihi"],
            "final": ["yarıyıl sonu sınavı"],
            "bütünleme": ["tek ders sınavı", "bütünleme sınavı"],
            "tarih": ["zaman", "başlangıç", "bitiş", "son tarih", "süre"],
            "kredi": ["ders kredisi", "AKTS"],
            "staj": ["zorunlu staj", "uygulamalı ders"],
            "burs": ["maddi yardım", "katkı"],
            "not": ["başarı notu", "harf notu"],
            "öğrenci belgesi": ["öğrenci kimliği", "transkript"]
        }
        concept_expansion = {
            "yatay geçiş": ["kurumlararası geçiş", "merkezi yerleştirme puanı"],
            "dikey geçiş": ["intibak", "dgs"],
            "disiplin": ["disiplin cezası", "uyarı", "kınama"],
            "sınav": ["ara sınav", "final", "bütünleme"],
            "mezuniyet": ["staj", "diploma", "bitirme projesi"],
            "tarih": ["başlangıç", "bitiş", "son tarih", "süre"],
            "ders kaydı": ["ekle/çıkar", "danışman onayı"],
            "başvuru": ["form", "dilekçe", "evrak"],
            "kayıt silme": ["ilişik kesme", "çıkış"]
        }
        hierarchical_expansion = {
            "ara sınav": ["sınav"],
            "final": ["sınav"],
            "bütünleme": ["sınav"],
            "kınama": ["disiplin"],
            "uyarı": ["disiplin"],
            "tarih": ["zaman"],
            "başvuru": ["belge"],
            "mezuniyet": ["öğrenci hakları"],
            "staj": ["mezuniyet"],
            "burs": ["öğrenci hakları"],
            "not": ["sınav"]
        }

        query = query.lower().replace("I", "ı").replace("İ", "i")
        original_tokens = self.turkish_preprocess(query)

        expanded = original_tokens.copy()

        for key in hint_expansion:
            if key in query:
                expanded.extend(hint_expansion[key][:max_hint])

        for t in original_tokens:
            if t in synonym_expansion:
                expanded.extend(synonym_expansion[t][:max_syn])

        for key in concept_expansion:
            if key in query:
                expanded.extend(concept_expansion[key][:max_concept])

        for t in original_tokens:
            if t in hierarchical_expansion:
                expanded.extend(hierarchical_expansion[t][:max_hier])

        expanded = self.unique_preserve_order(expanded)
        return " ".join(expanded)

    # -----------------------------
    # HYBRID RETRIEVAL
    # -----------------------------
    def hybrid_retrieve(self, query, top_k=100, alpha=0.5, title_boost=0.5):
        tokenized_query = self.turkish_preprocess(query)

        # BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

        # Dense scores
        query_emb = self.dense_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        dense_scores, _ = self.index.search(query_emb, len(self.dense_docs))
        dense_scores = dense_scores[0]
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-9)

        # Hybrid
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        # Title boost
        for i, title_tokens in enumerate(self.title_tokenized):
            if title_tokens:
                overlap = len(set(tokenized_query) & set(title_tokens))
                if overlap > 0:
                    hybrid_scores[i] += title_boost * (overlap / max(len(tokenized_query), 1))

        # Normalize again
        hybrid_scores = (hybrid_scores - np.min(hybrid_scores)) / (np.max(hybrid_scores) - np.min(hybrid_scores) + 1e-9)
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        top_scores = hybrid_scores[top_indices]
        return top_indices.tolist(), top_scores

    # -----------------------------
    # SCORE-WEIGHTED PRF
    # -----------------------------
    def weighted_prf(self, query, top_indices, top_scores, top_docs=10, top_terms=5):
        term_weights = {}
        for rank, idx in enumerate(top_indices[:top_docs]):
            tokens = self.turkish_preprocess(self.sparse_docs[idx])
            score_weight = top_scores[rank]
            for t in tokens:
                term_weights[t] = term_weights.get(t, 0) + score_weight
        feedback_terms = [t for t, _ in sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:top_terms]]
        return query + " " + " ".join(feedback_terms)

    # -----------------------------
    # CROSS-ENCODER RERANK
    # -----------------------------
    def cross_rerank(self, query, candidate_indices):
        pairs = [(query, self.dense_docs[idx]) for idx in candidate_indices]
        scores = self.cross_encoder.predict(pairs)
        return dict(zip(candidate_indices, scores))

    # -----------------------------
    # BUILD GOLD QUERY POOL
    # -----------------------------
    def build_pool(self):
        self.gold_queries = []
        self.pool_sizes = []

        for q_idx, query in tqdm(enumerate(self.queries), total=len(self.queries)):
            expanded_query = self.expand_query(query)
            tokenized_query = self.turkish_preprocess(expanded_query)

            # Candidate pools
            bm25_top = np.argsort(self.bm25.get_scores(tokenized_query))[::-1][:100]
            query_emb = self.dense_model.encode([expanded_query], convert_to_numpy=True)
            faiss.normalize_L2(query_emb)
            dense_scores, dense_indices = self.index.search(query_emb, 100)
            dense_top = dense_indices[0]
            hybrid_top, hybrid_scores = self.hybrid_retrieve(expanded_query, top_k=100)

            candidate_pool = list(set(bm25_top.tolist() + dense_top.tolist() + hybrid_top))
            self.pool_sizes.append(len(candidate_pool))

            # PRF + rerank
            prf_query = self.weighted_prf(expanded_query, hybrid_top, hybrid_scores)
            cross_scores = self.cross_rerank(prf_query, candidate_pool)
            final_sorted = sorted(candidate_pool, key=lambda x: cross_scores.get(x, 0), reverse=True)
            final_sorted = final_sorted[:self.max_pool_size]

            self.gold_queries.append({
                "query_id": int(self.query_ids[q_idx]),
                "query": query,
                "type": self.query_types[q_idx],
                "expanded_query": prf_query,
                "candidate_doc_ids": [self.doc_ids[i] for i in final_sorted]
            })

        print("Gold query pool built.")
        print("Avg pool size before cut:", np.mean(self.pool_sizes))

    # -----------------------------
    # SAVE POOL
    # -----------------------------
    def save(self, output_path="data/processed/gold_query_pool.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.gold_queries, f, ensure_ascii=False, indent=2)
        print(f"Saved gold query pool to {output_path}")
