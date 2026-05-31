import torch
import re
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from src.processing.preprocessing import turkish_preprocess


# ----------------------------------------------------------
# FILTERS
# ----------------------------------------------------------
QUERY_NOISE = {"nedir", "kaç", "hangisi", "kim", "ne", "nasıl"}

LOW_INFO = {
    "bir", "dir", "dır", "tir", "tır",
    "olan", "ile", "ve", "bu", "olarak",
    "için", "gibi", "her", "daha",
    "madde", "değer", "yön"
}


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def normalize_token(t):
    return re.sub(r"[^\wçğıöşü]", "", t.lower())


def is_valid_word(w):
    return (
        len(w) > 3
        and w not in LOW_INFO
        and w not in QUERY_NOISE
        and not w.endswith(("dir", "dır", "tir", "tır", "si", "ları", "leri"))
    )


# ==========================================================
# EXPLANATION ENGINE (FIXED)
# ==========================================================
class ExplanationEngine:
    def __init__(self, ir_system):
        self.ir = ir_system
        self.cross_encoder = ir_system.cross_encoder

        if ir_system.dense_model is not None:
            self.sent_model = ir_system.dense_model
        else:
            self.sent_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )

        self.cross_model = None
        self.tokenizer = None

        if self.cross_encoder is not None:
            self.cross_model = self.cross_encoder.model
            self.cross_model.eval()
            model_name = self.cross_model.config._name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================
    def explain_query(self, query, top_k=5, use_cross=True):
        doc_ids, _ = self.ir.retrieve(query=query, top_k=top_k, use_cross=use_cross)

        results = []

        for doc_id in doc_ids:
            doc_idx = self.ir.doc_id_to_index[doc_id]
            doc = self.ir.doc_json[doc_idx]
            text = doc.get("text", "")

            explanation = self.fuse_explanations(query, text)
            ig = explanation["ig"]

            phrases = self.extract_phrases(query, text, ig)

            faith = self.faithfulness_score(query, text, ig)
            suff = self.sufficiency_score(query, text, ig)
            cons = self.explanation_consistency(query, text)

            results.append({
                "doc_id": doc_id,
                "text": text,
                "explanation": explanation,
                "phrases": phrases,
                "faithfulness": faith,
                "sufficiency": suff,
                "consistency": cons
            })

        return results

    # ==========================================================
    # SENTENCE EXPLANATION (STRONG BASE)
    # ==========================================================
    def sentence_explanation(self, query, document, k=2):
        sentences = split_sentences(document)

        if not sentences:
            return document[:400]

        q_emb = self.sent_model.encode(query, convert_to_tensor=True)
        s_emb = self.sent_model.encode(sentences, convert_to_tensor=True)

        scores = util.cos_sim(q_emb, s_emb)[0]
        top_idx = torch.topk(scores, k=min(k, len(scores))).indices

        return " ".join([sentences[i] for i in top_idx])

    # ==========================================================
    # SMOOTH IG (FIXED TOKEN HANDLING)
    # ==========================================================
    def smooth_integrated_gradients(self, query, document, steps=20):

        if self.cross_model is None:
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.cross_model.to(device)
        tokenizer = self.tokenizer

        encoding = tokenizer(
            query,
            document,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        ).to(device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"][0]

        embeddings = model.get_input_embeddings()(input_ids)

        baseline = torch.zeros_like(embeddings)

        total_grads = torch.zeros_like(embeddings)

        for alpha in torch.linspace(0, 1, steps).to(device):
            scaled = baseline + alpha * (embeddings - baseline)
            scaled.requires_grad_(True)

            outputs = model(inputs_embeds=scaled, attention_mask=attention_mask)
            logits = outputs.logits

            score = logits[:, 1].sum() if logits.shape[-1] > 1 else logits.sum()

            grads = torch.autograd.grad(score, scaled)[0]
            total_grads += grads

        avg_grads = total_grads / steps
        ig = (embeddings - baseline) * avg_grads

        token_scores = ig.sum(dim=-1).squeeze().detach().cpu().numpy()

        return self._map_tokens_to_words(document, offsets, token_scores)

    # ==========================================================
    # FIXED TOKEN → WORD MAPPING
    # ==========================================================
    def _map_tokens_to_words(self, document, offsets, scores):

        word_scores = defaultdict(float)

        for (start, end), score in zip(offsets, scores):
            if start == end:
                continue

            word = document[start:end].lower()
            word = normalize_token(word)

            if is_valid_word(word):
                word_scores[word] += float(score)

        if not word_scores:
            return []

        # normalize
        max_val = max(abs(v) for v in word_scores.values()) + 1e-8

        return [(w, v / max_val) for w, v in word_scores.items()]

    # ==========================================================
    # SEMANTIC PHRASE EXTRACTION (FIXED)
    # ==========================================================
    def extract_phrases(self, query, document, ig_tokens, top_k=5):

        if not ig_tokens:
            return []

        # --------------------------------------------
        # CLEAN TOKENS
        # --------------------------------------------
        def is_clean(w):
            return (
                    len(w) > 3 and
                    w.isalpha() and
                    not w.endswith((
                        "dir", "dır", "tir", "tır",
                        "lar", "ler", "ları", "leri",
                        "da", "de", "ta", "te",
                        "si", "sı", "su", "sü",
                        "lu", "lü", "lı", "li",
                        "luk", "lük"
                    ))
            )

        # filter + normalize
        clean = [(w, abs(s)) for w, s in ig_tokens if is_clean(w)]

        if not clean:
            return []

        # --------------------------------------------
        # GROUP SIMILAR WORDS (VERY IMPORTANT)
        # --------------------------------------------
        # Example:
        # "öğrenci", "öğrencilerin", "öğrencinin" → "öğrenci"

        def stem_like(w):
            # simple Turkish heuristic stemmer
            for suf in ["leri", "ları", "lerin", "ların", "nin", "nın", "si", "sı", "de", "da"]:
                if w.endswith(suf):
                    return w[:-len(suf)]
            return w

        grouped = {}

        for w, s in clean:
            root = stem_like(w)
            if root not in grouped:
                grouped[root] = 0
            grouped[root] += s

        # --------------------------------------------
        # SORT BY IMPORTANCE
        # --------------------------------------------
        sorted_words = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

        # --------------------------------------------
        # EXPAND INTO REAL TERMS FROM DOCUMENT
        # --------------------------------------------
        results = []

        for root, _ in sorted_words:

            # find best matching phrase in document
            matches = re.findall(rf"\b\w*{root}\w*\b", document.lower())

            if matches:
                # pick longest meaningful form
                best = max(matches, key=len)
                results.append(best)

        # deduplicate
        final = []
        seen = set()

        for r in results:
            if r not in seen:
                final.append(r)
                seen.add(r)

        return final[:top_k]

    # ==========================================================
    # METRICS (FIXED)
    # ==========================================================
    def faithfulness_score(self, query, document, ig_tokens, k=10):

        if self.cross_encoder is None or ig_tokens is None:
            return 0.0

        important = sorted(ig_tokens, key=lambda x: abs(x[1]), reverse=True)[:k]
        words = set([t for t, _ in important])

        def remove(text):
            tokens = turkish_preprocess(text, False)
            return " ".join([w for w in tokens if w not in words])

        orig = self.cross_encoder.predict([(query, document)])[0]
        masked = self.cross_encoder.predict([(query, remove(document))])[0]

        return float((orig - masked) / (abs(orig) + 1e-8))

    # ✅ FIXED: no longer destroys sentence
    def sufficiency_score(self, query, document, ig_tokens, k=10):

        important = sorted(ig_tokens, key=lambda x: abs(x[1]), reverse=True)[:k]
        words = set([t for t, _ in important])

        sentences = split_sentences(document)

        kept = [
            s for s in sentences
            if any(w in s.lower() for w in words)
        ]

        if not kept:
            return 0.0

        kept_text = " ".join(kept)

        orig = self.cross_encoder.predict([(query, document)])[0]
        new = self.cross_encoder.predict([(query, kept_text)])[0]

        return float(new / (abs(orig) + 1e-8))

    def explanation_consistency(self, query, document):

        ig_tokens = self.smooth_integrated_gradients(query, document)
        sentence = self.sentence_explanation(query, document)

        if ig_tokens is None:
            return 0.0

        sentence_words = set(turkish_preprocess(sentence))
        top_ig = sorted(ig_tokens, key=lambda x: abs(x[1]), reverse=True)[:10]
        ig_words = set([t for t, _ in top_ig])

        return len(sentence_words & ig_words) / (len(ig_words) + 1e-8)

    # ==========================================================
    # FUSION
    # ==========================================================
    def fuse_explanations(self, query, document):
        return {
            "sentence": self.sentence_explanation(query, document),
            "ig": self.smooth_integrated_gradients(query, document)
        }