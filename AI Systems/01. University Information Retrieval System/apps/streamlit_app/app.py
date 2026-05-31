import os
import sys
import time
import logging
import random
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import faiss

from google import genai
from google.genai import types
import os

client = genai.Client(api_key=GOOGLE_API_KEY)

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.utils.load_ir import load_university_ir
from src.processing import query_expansion
from src.explanations.explanation_engine import ExplanationEngine
from src.explanations.visualization import plot_ig_heatmap

# ==========================================================
# FIXED MODEL + BEST CONFIG
# ==========================================================
IR_MODEL_CUSTOM_PATH = ROOT_DIR / "models/ir_model_custom_ce"

BEST_CONFIG = {
    "expansion": False,
    "meta": True,
    "title": True,
    "length": False,
    "cross": True,
    "use_dense": True,
    "alpha": 0.5
}

MODEL_CONFIGS = {
    "Best (Hybrid + Cross)": {
        **BEST_CONFIG
    },
    "Hybrid (no Cross)": {
        **BEST_CONFIG,
        "cross": False
    },
    "BM25 Only": {
        **BEST_CONFIG,
        "use_dense": False,
        "cross": False
    },
    "Dense Only": {
        **BEST_CONFIG,
        "use_dense": True,
        "cross": False,
        "alpha": 1.0  # fully dense
    }
}
PREVIEW_CHARS = 100


# ==========================================================
# ENVIRONMENT SETUP
# ==========================================================
def setup_environment(seed=42):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    logging.basicConfig(level=logging.ERROR)

    for lib in ["sentence_transformers", "transformers", "faiss"]:
        logging.getLogger(lib).setLevel(logging.ERROR)

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


setup_environment()

# ==========================================================
# STREAMLIT CONFIG
# ==========================================================
st.set_page_config(page_title="University IR", page_icon="🎓", layout="wide")

st.title("🎓 University Information Retrieval")
st.markdown("🚀 Hybrid + Meta + Title + Cross (Fixed Best Model)")

if "confirm_heavy" not in st.session_state:
    st.session_state.confirm_heavy = False

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
with st.sidebar:
    st.title("⚙️ Settings")

    st.success("🚀 Fixed Model: Hybrid + Meta + Title + Cross")

    with st.expander("Search Mode", expanded=False):
        use_rag = st.checkbox("Enable RAG (gemini-3-flash-preview Answer)", value=False)

    with st.expander("Retrieval Settings", expanded=False):
        top_k = st.slider("Top-K Results", 1, 50, 3)
        suggest_reformulations = st.checkbox("Suggest Query Reformulations", value=False)
        show_scores = st.checkbox("Show Detailed Scores", value=True)

    with st.expander("Explanations", expanded=False):
        enable_explanations = st.checkbox("Enable Explanations", value=True)
        explanation_depth = st.selectbox(
            "Explanation Detail",
            ["Fast", "Full (IG + Metrics)"]
        )
    with st.expander("Advanced", expanded=False):
        with st.expander("Model Variants", expanded=False):
            model_choice = st.selectbox(
                "Choose Retrieval Model",
                [
                    "Best (Hybrid + Cross)",
                    "Hybrid (no Cross)",
                    "BM25 Only",
                    "Dense Only"
                ]
            )
        with st.expander("Compare Models", expanded=False):
            compare_mode = st.checkbox("Enable Comparison Mode", value=False)
            model_a = st.selectbox(
                "Model A",
                list(MODEL_CONFIGS.keys()),
                index=0,
                key="model_a"
            )

            model_b = st.selectbox(
                "Model B",
                list(MODEL_CONFIGS.keys()),
                index=1,
                key="model_b"
            )

enable_full_explanations = (
    enable_explanations and explanation_depth == "Full (IG + Metrics)"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def get_model(folder_path):
    return load_university_ir(folder=folder_path)


if "model_loaded_shown" not in st.session_state:
    st.session_state.model_loaded_shown = False

status_placeholder = st.sidebar.empty()

with st.spinner("Loading IR model..."):
    ir_model = get_model(str(IR_MODEL_CUSTOM_PATH))

if not st.session_state.model_loaded_shown:
    status_placeholder.success("Model loaded!")
    st.session_state.model_loaded_shown = True


# ==========================================================
# INIT EXPLAINER
# ==========================================================
@st.cache_resource
def get_explainer(_ir_model):
    return ExplanationEngine(_ir_model)


explainer = get_explainer(ir_model)

# ==========================================================
# SESSION STATE
# ==========================================================
if "query" not in st.session_state:
    st.session_state.query = ""

if "selected_reformulation" not in st.session_state:
    st.session_state.selected_reformulation = False


# ==========================================================
# QUERY PREPROCESS
# ==========================================================
def preprocess_query(query: str):
    return query_expansion.expand_query_weighted(query)


# ==========================================================
# RAG HELPERS
# ==========================================================
def build_context(doc_ids, max_chars=15000):
    context = []
    total_len = 0

    for doc_id in doc_ids:
        doc_idx = ir_model.doc_id_to_index[doc_id]
        text = ir_model.doc_json[doc_idx].get("text", "")

        if total_len + len(text) > max_chars:
            break

        context.append(text)
        total_len += len(text)

    return "\n\n".join(context)


def generate_answer(query, context):
    # 2. Use client.models.generate_content
    # Note: System instructions are now part of the 'config'
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"Context:\n{context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert assistant for university regulations.\n"
                "Answer the question strictly based on the provided context.\n\n"
                "Rules:\n"
                "- Do NOT use any external knowledge.\n"
                "- If the answer cannot be found, respond exactly: 'Not found in documents'.\n"
                "- Be concise, precise, and formal.\n"
                "- Use terminology exactly as written in the documents.\n"
                "- When possible, support your answer with short phrases from the context.\n"
                "- Do not make assumptions or infer beyond the text.\n\n"
                "Important:\n"
                "- The provided context is ordered by relevance.\n"
                "- Give higher importance to earlier documents.\n"
                "- Prioritize information from top-ranked documents over lower-ranked ones.\n"
            ),
            temperature=0.2
        )
    )

    return response.text


def run_retrieval(config, query_text):
    return ir_model.retrieve_ablation(
        query=query_text,
        top_k=top_k,
        use_expansion=config["expansion"],
        use_meta=config["meta"],
        use_title=config["title"],
        use_length=config["length"],
        use_cross=config["cross"],
        use_dense=config["use_dense"],
        alpha=config["alpha"]
    )


def compute_overlap_and_diff(results_a, results_b):
    set_a = set(results_a)
    set_b = set(results_b)

    overlap = set_a & set_b

    rank_diff = {}
    for doc in overlap:
        rank_a = results_a.index(doc) + 1
        rank_b = results_b.index(doc) + 1
        rank_diff[doc] = rank_a - rank_b

    return overlap, rank_diff


# ==========================================================
# DISPLAY RESULTS (UNCHANGED)
# ==========================================================
def display_results(doc_ids, query_text=None, explanations=None, use_cross=True):
    # -------------------------
    # Precompute query embedding
    # -------------------------
    query_emb = None
    if ir_model.dense_model is not None and ir_model.index is not None:
        query_emb = ir_model.dense_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

    # -------------------------
    # BM25 (RAW + NORMALIZED)
    # -------------------------
    bm25_raw = None
    bm25_scores = None

    if show_scores and query_text:
        bm25_raw = ir_model.bm25.get_scores(query_text.lower().split())
        bm25_min = np.min(bm25_raw)
        bm25_max = np.max(bm25_raw)
        bm25_scores = (bm25_raw - bm25_min) / (bm25_max - bm25_min + 1e-9)

    # -------------------------
    # DENSE (RAW + NORMALIZED)
    # -------------------------
    dense_raw = np.zeros(ir_model.num_docs)
    dense_scores = None

    if show_scores and query_emb is not None:
        try:
            for i in range(ir_model.num_docs):
                doc_emb = ir_model.index.reconstruct(i).reshape(1, -1)
                faiss.normalize_L2(doc_emb)
                dense_raw[i] = float(np.dot(query_emb, doc_emb.T)[0][0])
        except Exception as e:
            logging.warning(f"Dense scoring failed: {e}")

        d_min = np.min(dense_raw)
        d_max = np.max(dense_raw)
        dense_scores = (dense_raw - d_min) / (d_max - d_min + 1e-9)

    # -------------------------
    # CROSS ENCODER (RAW + NORMALIZED)
    # -------------------------
    ce_raw = None
    ce_scores = None

    if show_scores and use_cross and ir_model.cross_encoder is not None:
        texts = [
            ir_model.doc_json[ir_model.doc_id_to_index[doc_id]].get("text", "")
            for doc_id in doc_ids
        ]

        ce_raw = ir_model.cross_encoder.predict([(query_text, t) for t in texts])
        ce_raw = np.array(ce_raw).flatten()

        ce_min = np.min(ce_raw)
        ce_max = np.max(ce_raw)
        ce_scores = (ce_raw - ce_min) / (ce_max - ce_min + 1e-9)

    # -------------------------
    # DISPLAY LOOP
    # -------------------------
    for rank, doc_id in enumerate(doc_ids, start=1):
        doc_idx = ir_model.doc_id_to_index[doc_id]
        doc = ir_model.doc_json[doc_idx]

        text_full = doc.get("text", "")
        text_preview = text_full[:PREVIEW_CHARS]
        if len(text_full) > PREVIEW_CHARS:
            text_preview += "..."

        exp = explanations[rank - 1] if explanations else None

        with st.container():
            col_rank, col_content = st.columns([1, 8])

            # Rank
            with col_rank:
                st.markdown(f"## {rank}")
                if rank == 1:
                    st.success("🏆")
                elif rank <= 3:
                    st.info("🔝")

            # Content
            with col_content:
                doc_title = doc.get("regulation_title", None)

                if doc_title:
                    st.markdown(f"#### 📄 {doc_title}")
                    st.caption(f"ID: `{doc_id}`")
                else:
                    st.markdown(f"### 📄 Document `{doc_id}`")

                st.write(text_preview)

                with st.expander("View Full Document"):
                    st.write(text_full)

                # -------------------------
                # SCORES DISPLAY
                # -------------------------
                if show_scores and query_text:

                    # BM25
                    bm25_norm = float(bm25_scores[doc_idx]) if bm25_scores is not None else 0.0
                    bm25_orig = float(bm25_raw[doc_idx]) if bm25_raw is not None else 0.0

                    # Dense
                    dense_norm = float(dense_scores[doc_idx]) if dense_scores is not None else 0.0
                    dense_orig = float(dense_raw[doc_idx]) if dense_raw is not None else 0.0

                    # CE (note: CE is only for top-k docs)
                    ce_norm = float(ce_scores[rank - 1]) if ce_scores is not None else 0.0
                    ce_orig = float(ce_raw[rank - 1]) if ce_raw is not None else 0.0

                    c1, c2, c3 = st.columns(3)

                    # BM25
                    c1.metric("BM25", f"{bm25_norm:.3f}")
                    c1.caption(f"raw: {bm25_orig:.2f}")

                    # Dense
                    c2.metric("Dense", f"{dense_norm:.3f}")
                    c2.caption(f"raw: {dense_orig:.3f}")

                    # CE
                    c3.metric("CE", f"{ce_norm:.3f}")
                    c3.caption(f"raw: {ce_orig:.3f}")

                # -------------------------
                # EXPLANATIONS
                # -------------------------
                if enable_full_explanations and exp is not None:

                    st.markdown("#### 🧠 Why this result?")

                    key_sentence = exp["explanation"]["sentence"]
                    phrases = exp.get("phrases", [])

                    simple_reason = "Matches key content related to your query."
                    if phrases:
                        simple_reason += f" Focus on: {', '.join(phrases[:3])}"

                    st.info(simple_reason)

                    with st.expander("🔬 Show detailed explanation"):

                        st.markdown("**Most Relevant Sentence:**")
                        st.write(key_sentence)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Faithfulness", f"{exp['faithfulness']:.3f}")
                        col2.metric("Sufficiency", f"{exp['sufficiency']:.3f}")
                        col3.metric("Consistency", f"{exp['consistency']:.3f}")

                        if phrases:
                            st.markdown("**Key Concepts:**")
                            for p in phrases[:5]:
                                st.markdown(f"- `{p}`")

                        st.caption(
                            "Faithfulness: impact of tokens on prediction. "
                            "Sufficiency: whether key tokens alone explain relevance. "
                            "Consistency: agreement across explanation methods."
                        )

                        fig = plot_ig_heatmap(exp)
                        st.pyplot(fig)

            st.divider()


# ==========================================================
# MAIN INPUT
# ==========================================================
query = st.text_input(
    "Enter your query:",
    key="query",
    placeholder="e.g., university ranking Europe AI programs"
)
search_button = st.button("Search")

# ==========================================================
# SEARCH STATE
# ==========================================================
if "run_search" not in st.session_state:
    st.session_state.run_search = False

if search_button:
    st.session_state.run_search = True

# ==========================================================
# SEARCH FLOW
# ==========================================================
if search_button:
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    preprocessed = preprocess_query(query)
    query_text = " ".join(preprocessed["expanded"]) or query

    # ==========================================================
    # QUERY REFORMULATIONS
    # ==========================================================
    reformulations = None

    if suggest_reformulations:
        with st.spinner("Generating reformulation suggestions..."):

            reformulations = ir_model.suggest_query_reformulations_ranked(
                query=query,
                top_k_docs=5,
                prf_terms=5,
                max_suggestions=5,
                use_cross=False
            )

    # ==========================================================
    # INFO
    # ==========================================================
    if not compare_mode:
        st.info(
            f"Query used: `{query_text}` | "
            f"{'Mode: RAG enabled' if use_rag else 'Mode: Retrieval'} | "
            f"Model: {model_choice}"
        )
    else:
        st.info(f"Query used: `{query_text}` | Compare: {model_a} vs {model_b}")
    st.markdown("---")

    # ==========================================================
    # SHOW REFORMULATIONS
    # ==========================================================
    if suggest_reformulations and reformulations:

        st.markdown("## 🔍 Suggested Reformulations")
        st.caption("Alternative queries that may improve retrieval quality")

        for i, (suggestion, score) in enumerate(reformulations, start=1):

            col1, col2 = st.columns([8, 1])

            with col1:
                if st.button(
                        suggestion,
                        key=f"reform_{i}"
                ):
                    st.session_state.query = suggestion
                    st.session_state.selected_reformulation = True
                    st.session_state.run_search = True
                    st.rerun()

            with col2:
                st.metric("Score", f"{score:.3f}")

        st.divider()

    # ==========================================================
    # STOP AFTER SHOWING SUGGESTIONS
    # ==========================================================
    if suggest_reformulations and not st.session_state.selected_reformulation:
        st.session_state.run_search = False
        st.stop()

    # ==========================================================
    # RETRIEVAL
    # ==========================================================
    if compare_mode:
        config_a = MODEL_CONFIGS[model_a]
        config_b = MODEL_CONFIGS[model_b]

        results_a = run_retrieval(config_a, query_text)
        results_b = run_retrieval(config_b, query_text)

        overlap, rank_diff = compute_overlap_and_diff(results_a, results_b)

    else:
        config = MODEL_CONFIGS[model_choice]
        results = run_retrieval(config, query_text)

    explanations = None
    if enable_full_explanations:
        explanations = explainer.explain_query(
            query=query,
            top_k=top_k,
            use_cross=BEST_CONFIG["cross"]
        )

    # ==========================================================
    # FINAL SWITCH
    # ==========================================================
    if compare_mode:
        st.markdown("## 🆚 Model Comparison")

        st.caption(f"Common documents: {len(overlap)} / {top_k}")

        if len(overlap) == 0:
            st.caption("Models retrieve completely different documents.")
        elif len(overlap) < top_k / 2:
            st.caption("Models partially agree but differ significantly.")
        else:
            st.caption("Models largely agree on relevant documents.")

        col1, col2 = st.columns([1, 1], gap="large")

        # ======================
        # MODEL A
        # ======================
        with col1:
            with st.container():
                st.markdown(f"### 🔵 {model_a}")

                if not use_rag:
                    display_results(results_a, query_text, None)
                else:
                    context_a = build_context(results_a)
                    answer_a = generate_answer(query, context_a)
                    st.write(answer_a)

        # ======================
        # MODEL B
        # ======================
        with col2:
            with st.container():
                st.markdown(f"### 🟢 {model_b}")

                if not use_rag:
                    display_results(results_b, query_text, None)
                else:
                    context_b = build_context(results_b)
                    answer_b = generate_answer(query, context_b)
                    st.write(answer_b)

        # ======================
        # RANK DIFFERENCE VIEW
        # ======================
        st.divider()
        st.markdown("## 📊 Ranking Differences")

        if len(overlap) == 0:
            st.warning("No common documents in top-k results.")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 🔵 Unique to Model A")
                unique_a = [doc for doc in results_a if doc not in results_b]

                if len(unique_a) == 0:
                    st.caption("No unique documents")
                else:
                    for doc in unique_a:
                        st.markdown(f"{'&nbsp;' * 14}• {doc}", unsafe_allow_html=True)
            with colB:
                st.markdown("### 🟢 Unique to Model B")
                unique_b = [doc for doc in results_b if doc not in results_a]

                if len(unique_b) == 0:
                    st.caption("No unique documents")
                else:
                    for doc in unique_b:
                        st.markdown(f"{'&nbsp;' * 14}• {doc}", unsafe_allow_html=True)

        else:
            for doc in overlap:
                rank_a = results_a.index(doc) + 1
                rank_b = results_b.index(doc) + 1
                diff = rank_a - rank_b

                if diff == 0:
                    st.write(f"= Doc {doc} same rank ({rank_a})")
                elif diff > 0:
                    st.write(f"🔼 Doc {doc} higher in Model B (A:{rank_a} → B:{rank_b})")
                else:
                    st.write(f"🔽 Doc {doc} higher in Model A (A:{rank_a} → B:{rank_b})")

    else:
        context = build_context(results)

        if not use_rag:
            display_results(results, query_text, explanations)

        else:
            with st.spinner("Generating answer with gemini-3-flash-preview ..."):
                answer = generate_answer(query, context)

            st.markdown("## 🤖 Answer")
            st.write(answer)

            with st.expander("📄 Sources"):
                display_results(results, query_text, explanations)

    # ==========================================================
    # RESET SEARCH STATE
    # ==========================================================
    st.session_state.run_search = False
    st.session_state.selected_reformulation = False
