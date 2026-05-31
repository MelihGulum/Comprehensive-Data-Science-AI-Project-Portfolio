import numpy as np
from src.processing.preprocessing import turkish_preprocess
import re
from src.processing.query_expansion import expand_query_weighted
import faiss


# -----------------------------
# UNIVERSITY IR CLASS
# -----------------------------
class UniversityIR:
    def __init__(self,
                 doc_json,
                 sparse_docs,
                 dense_docs,
                 title_tokenized,
                 bm25_model,
                 dense_model,
                 faiss_index,
                 cross_encoder=None,
                 es_client=None,
                 es_index=None,
                 best_alpha=0.5,
                 alpha_by_type=None,
                 best_weights=None):

        self.doc_json = doc_json
        self.doc_ids = [d["doc_id"] for d in doc_json]
        self.doc_id_to_index = {d["doc_id"]: i for i, d in enumerate(doc_json)}

        self.sparse_docs = sparse_docs
        self.dense_docs = dense_docs
        self.title_tokenized = title_tokenized
        self.num_docs = len(self.doc_ids)

        self.bm25 = bm25_model
        self.dense_model = dense_model
        self.index = faiss_index
        self.cross_encoder = cross_encoder
        self.es = es_client
        self.es_index = es_index

        # --- Learned parameters ---
        self.best_alpha = best_alpha
        self.alpha_by_type = alpha_by_type or {"default": best_alpha}
        self.best_weights = best_weights or {
            "expansion": 0.4,
            "prf": 0.3,
            "title_boost": 0.5,
            "meta_boost": 0.1
        }
        self.weights_by_type = {
            "default": (0.6, 0.25, 0.15),
            "short_query": (0.5, 0.4, 0.1),
            "legal_query": (0.8, 0.1, 0.1),
            "informational": (0.4, 0.5, 0.1),
            "process_query": (0.5, 0.3, 0.2),
            "date_query": (0.7, 0.1, 0.2),
        }

        # Convenience variables for title/meta boost
        self.title_boost = self.best_weights.get("title_boost", 0.5)
        self.meta_boost = self.best_weights.get("meta_boost", 0.1)

    # --------------------
    # Helper normalization
    # --------------------
    def _normalize(self, arr):
        arr = np.array(arr, dtype=float)

        # numerical stability
        arr = arr - np.max(arr)

        exp_arr = np.exp(arr)
        sum_exp = np.sum(exp_arr)

        if sum_exp < 1e-9:
            return np.zeros_like(arr)

        return exp_arr / sum_exp

    # --------------------
    # BM25 only
    # --------------------
    def bm25_only(self, query, top_k=10):
        tokenized_query = turkish_preprocess(query)
        scores = self._normalize(self.bm25.get_scores(tokenized_query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.doc_ids[int(i)] for i in top_indices]

    # --------------------
    # BM25 weighted
    # --------------------
    def bm25_weighted(self, original_tokens, expanded_tokens, w_orig=1.0, w_exp=0.4):
        scores = np.zeros(self.num_docs, dtype=float)

        for term in original_tokens:
            term_scores = self.bm25.get_scores([term])
            scores += w_orig * np.array(term_scores)

        for term in expanded_tokens:
            term_scores = self.bm25.get_scores([term])
            scores += w_exp * np.array(term_scores)

        return self._normalize(scores)

    def classify_query(self, query):
        q = query.lower()

        # ---- Rule-based classification ----
        if re.search(r"\b\d{4}\b", q):
            return "date_query"

        if re.search(r"\b(madde|fıkra|bent|article)\b", q):
            return "legal_query"

        tokens = turkish_preprocess(q)

        if len(tokens) <= 2:
            return "short_query"

        if any(t in ["nasıl", "nedir", "ne", "kaç", "hangi"] for t in tokens):
            return "informational"

        if any(t in ["başvuru", "kayıt", "mezuniyet", "staj"] for t in tokens):
            return "process_query"

        return "default"

    # --------------------
    # Hybrid retrieve
    # --------------------
    def hybrid_retrieve(self, query, top_k=100,
                        use_dense=True, use_elastic=True):

        tokenized_query = turkish_preprocess(query)
        expanded = expand_query_weighted(query)
        original_tokens = expanded["original"]
        expanded_tokens = expanded["expanded"]

        # Determine alpha based on query type
        qtype = self.classify_query(query)
        alpha = self.alpha_by_type.get(qtype, self.best_alpha)
        expansion_weight = self.best_weights.get("expansion", 0.4)

        if qtype == "short_query":
            # use_dense = True
            # use_elastic = False
            expansion_weight *= 0.7

        elif qtype == "legal_query":
            # use_dense = False
            # use_elastic = True
            expansion_weight *= 0.3

        elif qtype == "process_query":
            expansion_weight *= 1.3

        elif qtype == "informational":
            expansion_weight *= 1.1

        # Long query fix
        avg_token_len = len(query) / max(len(tokenized_query), 1)
        if len(tokenized_query) > 8 or avg_token_len > 7:
            expansion_weight *= 0.5

        # BM25
        bm25_scores = self.bm25_weighted(original_tokens, expanded_tokens, w_orig=1.0, w_exp=expansion_weight)

        # Dense
        dense_scores = np.zeros(self.num_docs)
        if use_dense and self.index is not None:
            query_embedding = self.dense_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            scores_raw, indices = self.index.search(query_embedding, min(self.num_docs, self.index.ntotal))
            for score, idx in zip(scores_raw[0], indices[0]):
                if 0 <= int(idx) < self.num_docs:
                    dense_scores[int(idx)] = float(score)
            dense_scores = self._normalize(dense_scores)

        # Elasticsearch
        elastic_scores = np.zeros(self.num_docs)
        if use_elastic and self.es is not None and self.es_index is not None:
            body = {
                "query": {"multi_match": {"query": query,
                                          "fields": ["content^2", "title^3", "category", "faculty"]}},
                "size": self.num_docs
            }
            response = self.es.search(index=self.es_index, body=body)
            for hit in response["hits"]["hits"]:
                doc_id = hit["_source"]["doc_id"]
                if doc_id in self.doc_id_to_index:
                    elastic_scores[self.doc_id_to_index[doc_id]] = float(hit["_score"])
            elastic_scores = self._normalize(elastic_scores)

        # Combine scores
        hybrid_scores = bm25_scores.copy()
        qtype = self.classify_query(query)
        w_bm25, w_dense, w_elastic = self.weights_by_type.get(qtype, self.weights_by_type["default"])
        total = w_bm25 + w_dense + w_elastic
        if total > 0:
            w_bm25, w_dense, w_elastic = w_bm25 / total, w_dense / total, w_elastic / total

        hybrid_scores = np.zeros(self.num_docs)

        if use_dense:
            hybrid_scores += w_dense * dense_scores

        if use_elastic:
            hybrid_scores += w_elastic * elastic_scores

        hybrid_scores += w_bm25 * bm25_scores

        # Title boost
        for i, title_tokens in enumerate(self.title_tokenized):
            overlap = len(set(tokenized_query) & set(title_tokens))
            if overlap > 0:
                hybrid_scores[i] += self.title_boost * (overlap / max(len(tokenized_query), 1))

        # Meta boost
        for i, d in enumerate(self.doc_json):
            meta_match = 0
            for field in ["article_type", "article_no", "paragraph_no", "clause"]:
                if field in d and d[field] != "-":
                    if re.search(rf"\b{re.escape(str(d[field]))}\b", query):
                        meta_match += 1
            for field in ["category", "faculty", "degree_level"]:
                if field in d:
                    if re.search(rf"\b{re.escape(d[field].lower())}\b", query.lower()):
                        meta_match += 1
            hybrid_scores[i] += self.meta_boost * meta_match

        # Length penalty
        for i in range(self.num_docs):
            length_penalty = min(len(self.sparse_docs[i].split()) / 200, 1)
            hybrid_scores[i] *= (0.8 + 0.2 * length_penalty)

        hybrid_scores = self._normalize(hybrid_scores)
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return top_indices.tolist(), hybrid_scores[top_indices]

    # --------------------
    # retrieve_ablation with learned weights
    # --------------------
    def retrieve_ablation(self, query, top_k=10,
                          use_expansion=True,
                          expansion_weight=None,
                          use_meta=True,
                          use_title=True,
                          use_length=True,
                          use_cross=False,
                          use_dense=True,
                          use_elastic=True,
                          alpha=None):

        expansion_weight = expansion_weight if expansion_weight is not None else self.best_weights.get("expansion", 0.4)
        # Determine alpha dynamically
        tokenized_query = turkish_preprocess(query)
        if alpha is None:
          qtype = self.classify_query(query)
          alpha = self.alpha_by_type.get(qtype, self.best_alpha)

        if use_expansion:
            expanded = expand_query_weighted(query)
            original_tokens = expanded["original"]
            expanded_tokens = expanded["expanded"]
        else:
            original_tokens = tokenized_query
            expanded_tokens = []

        bm25_scores = self.bm25_weighted(original_tokens, expanded_tokens, w_orig=1.0, w_exp=expansion_weight)
        dense_scores = np.zeros(self.num_docs)
        if use_dense and self.index is not None and self.dense_model is not None:
            query_embedding = self.dense_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            scores_raw, indices = self.index.search(query_embedding, min(self.num_docs, self.index.ntotal))
            for score, idx in zip(scores_raw[0], indices[0]):
                if 0 <= int(idx) < self.num_docs:
                    dense_scores[int(idx)] = float(score)
            dense_scores = self._normalize(dense_scores)

        elastic_scores = np.zeros(self.num_docs)
        if use_elastic and self.es is not None and self.es_index is not None:
            body = {"query": {"multi_match": {"query": query,
                                              "fields": ["content^2", "title^3", "category", "faculty"]}},
                    "size": self.num_docs}
            response = self.es.search(index=self.es_index, body=body)
            for hit in response["hits"]["hits"]:
                doc_id = hit["_source"]["doc_id"]
                if doc_id in self.doc_id_to_index:
                    elastic_scores[self.doc_id_to_index[doc_id]] = float(hit["_score"])
            elastic_scores = self._normalize(elastic_scores)


        hybrid_scores = bm25_scores.copy()
        qtype = self.classify_query(query)
        w_bm25, w_dense, w_elastic = self.weights_by_type.get(qtype, self.weights_by_type["default"])

        # normalize
        total = w_bm25 + w_dense + w_elastic
        if total > 0:
            w_bm25, w_dense, w_elastic = w_bm25 / total, w_dense / total, w_elastic / total

        hybrid_scores = np.zeros(self.num_docs)

        if use_dense:
            hybrid_scores += w_dense * dense_scores

        if use_elastic:
            hybrid_scores += w_elastic * elastic_scores

        hybrid_scores += w_bm25 * bm25_scores

        # Title boost
        if use_title:
            for i, title_tokens in enumerate(self.title_tokenized):
                overlap = len(set(tokenized_query) & set(title_tokens))
                if overlap > 0:
                    hybrid_scores[i] += self.title_boost * (overlap / max(len(tokenized_query), 1))

        # Meta boost
        if use_meta:
            for i, d in enumerate(self.doc_json):
                meta_match = 0
                for field in ["article_type", "article_no", "paragraph_no", "clause"]:
                    if field in d and d[field] != "-":
                        if re.search(rf"\b{re.escape(str(d[field]))}\b", query):
                            meta_match += 1
                for field in ["category", "faculty", "degree_level"]:
                    if field in d:
                        if re.search(rf"\b{re.escape(d[field].lower())}\b", query.lower()):
                            meta_match += 1
                hybrid_scores[i] += self.meta_boost * meta_match

        # Length penalty
        if use_length:
            for i in range(self.num_docs):
                length_penalty = min(len(self.sparse_docs[i].split()) / 200, 1)
                hybrid_scores[i] *= (0.8 + 0.2 * length_penalty)

        hybrid_scores = self._normalize(hybrid_scores)
        top_indices = np.argsort(hybrid_scores)[::-1][:min(50, self.num_docs)]

        if use_cross and self.cross_encoder is not None:
            cross_scores = self.cross_rerank(
                query,
                top_indices,
                hybrid_scores=hybrid_scores,
                qtype=self.classify_query(query),
                lambda_ce=0.7
            )
            final_sorted = sorted(top_indices, key=lambda x: cross_scores.get(int(x), 0.0), reverse=True)
        else:
            final_sorted = top_indices

        safe = [int(i) for i in final_sorted if 0 <= int(i) < self.num_docs]
        return [self.doc_ids[i] for i in safe[:top_k]]

    # --------------------
    # Cross-encoder rerank
    # --------------------
    def cross_rerank(self, query, candidate_indices, hybrid_scores=None, qtype=None, lambda_ce=0.7, batch_size=32):
        if self.cross_encoder is None:
            return {int(i): 0.0 for i in candidate_indices}

        # -------------------------
        # Query-aware lambda
        # -------------------------
        if qtype:
            if qtype == "legal_query":
                lambda_ce = 0.3
            elif qtype == "informational":
                lambda_ce = 0.7
            else:
                lambda_ce = 0.5

        # -------------------------
        # Prepare inputs
        # -------------------------
        pairs = []
        valid_indices = []

        for i in candidate_indices:
            if 0 <= int(i) < self.num_docs:
                text = self.doc_json[i]["text"]

                # truncate for speed
                text = " ".join(text.split()[:200])

                pairs.append((query, text))
                valid_indices.append(int(i))

        if not pairs:
            return {}

        # -------------------------
        # Predict
        # -------------------------
        scores = self.cross_encoder.predict(pairs, batch_size=batch_size)
        scores = np.array(scores)

        if len(scores.shape) == 2 and scores.shape[1] > 1:
            scores = scores[:, 1]

        scores = scores.flatten()

        # -------------------------
        # Normalize CE scores
        # -------------------------
        if scores.max() - scores.min() > 1e-9:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)

        # -------------------------
        # Fusion with hybrid
        # -------------------------
        final_scores = {}

        for idx, ce_score in zip(valid_indices, scores):
            if hybrid_scores is not None:
                h_score = hybrid_scores[idx]
                final_scores[idx] = lambda_ce * ce_score + (1 - lambda_ce) * h_score
            else:
                final_scores[idx] = ce_score

        return final_scores

    # -----------------------------
    # Hybrid PRF + Semantic Expansion Retrieval with learned weights
    # -----------------------------
    def retrieve(self, query, top_k=10,
                 prf_docs=5, prf_terms=5,
                 use_cross=True, w_prf=None, w_exp=None):

        # Use learned weights if not provided
        w_exp = w_exp if w_exp is not None else self.best_weights.get("expansion", 0.4) if self.best_weights else 0.4
        w_prf = w_prf if w_prf is not None else self.best_weights.get("prf", 0.3) if self.best_weights else 0.3

        # Step 1: Base semantic/weighted expansion
        expanded = expand_query_weighted(query)
        original_tokens = expanded["original"]
        semantic_tokens = expanded["expanded"]

        # Step 2: Retrieve top documents for PRF
        top_indices, _ = self.hybrid_retrieve(query, top_k=prf_docs)

        # Step 3: Collect PRF terms
        from collections import Counter
        prf_counter = Counter()
        for idx in top_indices:
            doc_tokens = turkish_preprocess(self.doc_json[idx]["text"])
            for t in doc_tokens:
                if t not in original_tokens and t not in semantic_tokens:
                    prf_counter[t] += 1
        top_prf_terms = [
                            t for t, c in prf_counter.most_common(prf_terms * 2)
                            if len(t) > 3 and c > 1
                        ][:prf_terms]

        # Step 4: Combine all tokens
        all_tokens = original_tokens + semantic_tokens + top_prf_terms

        # BM25 scores
        bm25_scores = self.bm25_weighted(original_tokens, semantic_tokens + top_prf_terms,
                                         w_orig=1.0, w_exp=w_exp)

        # Dense scores
        dense_scores = np.zeros(self.num_docs)
        if self.index is not None:
            query_embedding = self.dense_model.encode([" ".join(all_tokens)], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            scores_raw, indices = self.index.search(query_embedding, min(self.num_docs, self.index.ntotal))
            for score, idx in zip(scores_raw[0], indices[0]):
                if 0 <= int(idx) < self.num_docs:
                    dense_scores[int(idx)] = float(score)
            dense_scores = self._normalize(dense_scores)

        # Elasticsearch scores
        elastic_scores = np.zeros(self.num_docs)
        if self.es is not None and self.es_index is not None:
            body = {
                "query": {"multi_match": {"query": " ".join(all_tokens),
                                          "fields": ["content^2", "title^3", "category", "faculty"]}},
                "size": self.num_docs
            }
            response = self.es.search(index=self.es_index, body=body)
            for hit in response["hits"]["hits"]:
                doc_id = hit["_source"]["doc_id"]
                if doc_id in self.doc_id_to_index:
                    elastic_scores[self.doc_id_to_index[doc_id]] = float(hit["_score"])
            elastic_scores = self._normalize(elastic_scores)

        # Step 5: Determine alpha
        qtype = self.classify_query(query)
        alpha = self.alpha_by_type.get(qtype, self.best_alpha)

        # Step 6: Hybrid scoring
        hybrid_scores = bm25_scores.copy()
        qtype = self.classify_query(query)
        w_bm25, w_dense, w_elastic = self.weights_by_type.get(qtype, self.weights_by_type["default"])

        total = w_bm25 + w_dense + w_elastic
        if total > 0:
            w_bm25, w_dense, w_elastic = w_bm25 / total, w_dense / total, w_elastic / total

        hybrid_scores = np.zeros(self.num_docs)

        if self.index is not None:
            hybrid_scores += w_dense * dense_scores

        if self.es is not None:
            hybrid_scores += w_elastic * elastic_scores

        hybrid_scores += w_bm25 * bm25_scores

        # Title boost
        for i, title_tokens in enumerate(self.title_tokenized):
            overlap = len(set(original_tokens) & set(title_tokens))
            if overlap > 0:
                hybrid_scores[i] += self.title_boost * (overlap / max(len(original_tokens), 1))

        # Meta boost
        for i, d in enumerate(self.doc_json):
            meta_match = 0
            for field in ["article_type", "article_no", "paragraph_no", "clause"]:
                if field in d and d[field] != "-":
                    if re.search(rf"\b{re.escape(str(d[field]))}\b", query):
                        meta_match += 1
            for field in ["category", "faculty", "degree_level"]:
                if field in d:
                    if re.search(rf"\b{re.escape(d[field].lower())}\b", query.lower()):
                        meta_match += 1
            hybrid_scores[i] += self.meta_boost * meta_match

        # Length penalty
        for i in range(self.num_docs):
            length_penalty = min(len(self.sparse_docs[i].split()) / 200, 1)
            hybrid_scores[i] *= (0.8 + 0.2 * length_penalty)

        hybrid_scores = self._normalize(hybrid_scores)

        # Top-k indices
        rerank_k = min(100, self.num_docs)
        top_candidates = np.argsort(hybrid_scores)[::-1][:rerank_k]

        # Optional cross-encoder rerank
        if use_cross and self.cross_encoder is not None:
            qtype = self.classify_query(query)

            cross_scores = self.cross_rerank(
                query,
                top_candidates,
                hybrid_scores=hybrid_scores,
                qtype=qtype,
                lambda_ce=0.7
            )

            final_sorted = sorted(top_candidates, key=lambda x: cross_scores.get(int(x), 0.0), reverse=True)
        else:
            final_sorted = top_candidates

        safe = [int(i) for i in final_sorted if 0 <= int(i) < self.num_docs]
        scores_ordered = [hybrid_scores[i] for i in safe]
        return [self.doc_ids[i] for i in safe], scores_ordered

    # -----------------------------
    # Suggest query reformulations with learned weights
    # -----------------------------
    def suggest_query_reformulations_ranked(self, query,
                                            top_k_docs=5,
                                            prf_terms=5,
                                            max_suggestions=10,
                                            use_cross=True):
        # Base semantic/weighted expansion
        expanded = expand_query_weighted(query)
        original_tokens = expanded["original"]
        semantic_tokens = expanded["expanded"]

        # Retrieve top docs for PRF
        top_indices, _ = self.hybrid_retrieve(query, top_k=top_k_docs)

        # Collect PRF terms
        from collections import Counter
        prf_counter = Counter()
        for idx in top_indices:
            doc_tokens = turkish_preprocess(self.doc_json[idx]["text"])
            for t in doc_tokens:
                if t not in original_tokens and t not in semantic_tokens:
                    prf_counter[t] += 1
        top_prf_terms = [
                            t for t, c in prf_counter.most_common(prf_terms * 2)
                            if len(t) > 3 and c > 1
                        ][:prf_terms]

        # Generate candidate reformulated queries
        candidates = []

        # Single PRF-term suggestions
        for t in top_prf_terms:
            candidates.append(" ".join(original_tokens + [t]))

        # Multi-term PRF combinations (2-terms)
        from itertools import combinations
        for combo in combinations(top_prf_terms, 2):
            candidates.append(" ".join(original_tokens + list(combo)))

        # Semantic expansion suggestions
        for t in semantic_tokens:
            candidates.append(" ".join(original_tokens + [t]))

        # Remove duplicates
        candidates = list(dict.fromkeys(candidates))

        # Score candidate queries using expected retrieval quality
        query_scores = []
        for cand_query in candidates:
            top_docs, scores = self.retrieve(cand_query, top_k=5, use_cross=use_cross)
            expected_score = np.sum(scores) if len(scores) > 0 else 0.0
            query_scores.append((cand_query, expected_score))

        # Sort candidates by expected retrieval score
        query_scores.sort(key=lambda x: x[1], reverse=True)
        return query_scores[:max_suggestions]
