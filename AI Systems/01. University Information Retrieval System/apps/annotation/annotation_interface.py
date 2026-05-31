import streamlit as st
import json
import re
from pathlib import Path

# ======================
# STREAMLIT CONFIG
# ======================
st.set_page_config(page_title="Gold Pool Annotation", layout="wide")

# ======================
# DATASET PATHS
# ======================
DATASET_DIR = Path("data")
CORPUS_JSON = DATASET_DIR / "processed/corpus_preprocessed.json"
TRAIN_JSON = DATASET_DIR / "processed/qrels/qrel_train.json"
TEST_JSON = DATASET_DIR / "processed/qrels/qrel_test.json"
ANNOTATIONS_JSON = DATASET_DIR / "processed/qrels/annotations.json"

BATCH_SIZE = 10  # Adjust batch size for performance


# ======================
# CACHE LOADING
# ======================
@st.cache(allow_output_mutation=True)
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


doc_json = load_json(CORPUS_JSON)
doc_map = {d["doc_id"]: d for d in doc_json}

# ======================
# SELECT SPLIT
# ======================
split = st.sidebar.selectbox("Select Split", ["train", "test"])
split_json = TRAIN_JSON if split == "train" else TEST_JSON
dataset = load_json(split_json)

queries = dataset["queries"]
qrels = dataset["qrels"]

# Build query_id -> list of doc_ids
qrel_map = {}
for q in qrels:
    qid = q["query_id"]
    qrel_map.setdefault(qid, []).append(q["doc_id"])

# ======================
# SESSION STATE INIT
# ======================
if "annotations" not in st.session_state:
    if ANNOTATIONS_JSON.exists():
        loaded = load_json(ANNOTATIONS_JSON)
        st.session_state.annotations = {k: (v if v else "Unsure") for k, v in loaded.items()}
    else:
        st.session_state.annotations = {}

st.session_state.query_idx = st.session_state.get("query_idx", 0)
st.session_state.batch_start = st.session_state.get("batch_start", 0)


# ======================
# UTILITIES
# ======================
def highlight_terms(text: str, terms: list[str]) -> str:
    if not terms:
        return text
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)


def get_batch(items: list, start: int, batch_size: int) -> list:
    return items[start:start + batch_size]


# ======================
# QUERY SELECTION
# ======================
query_idx = st.sidebar.selectbox(
    "Select Query",
    range(len(queries)),
    index=st.session_state.query_idx,
    format_func=lambda i: f"{i} - {queries[i]['query']}"
)
st.session_state.query_idx = query_idx
query_data = queries[query_idx]
query_id = query_data["query_id"]
highlight_terms_list = query_data['query'].split()

# Progress
progress = (query_idx + 1) / len(queries)
st.sidebar.progress(progress)
st.sidebar.write(f"Query {query_idx + 1}/{len(queries)}")

st.title(f"Gold Pool Annotation Tool [{split.upper()}]")
st.header(f"Query [{query_id}]: {query_data['query']}")

candidate_doc_ids = qrel_map.get(query_id, [])
total_docs = len(candidate_doc_ids)
st.write(f"Total candidate documents: {total_docs}")

# ======================
# BATCHING
# ======================
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Previous Batch") and st.session_state.batch_start > 0:
        st.session_state.batch_start -= BATCH_SIZE
        if st.session_state.batch_start < 0:
            st.session_state.batch_start = 0

with col_next:
    if st.button("Next Batch ➡️") and st.session_state.batch_start + BATCH_SIZE < total_docs:
        st.session_state.batch_start += BATCH_SIZE

# Compute current batch
batch_start = st.session_state.batch_start
batch_doc_ids = candidate_doc_ids[batch_start: batch_start + BATCH_SIZE]
batch_end = batch_start + len(batch_doc_ids)

# ======================
# DISPLAY CANDIDATES
# ======================
options = ["Highly Relevant", "Relevant", "Not Relevant", "Unsure"]

for idx, doc_id in enumerate(batch_doc_ids):
    doc = doc_map.get(doc_id, {})
    st.markdown("---")
    st.subheader(f"{batch_start + idx + 1}. {doc.get('title', 'No Title')} (Doc ID: {doc_id})")
    st.text(
        f"Regulation:   {doc.get('regulation_title', '-')}"
        
        f"\nFaculty:      {doc.get('faculty', '-')}"
        
        f"\nDegree Level: {doc.get('degree_level', '-')}"
    )

    metadata = []
    if "type" in doc: metadata.append(f"Type: {doc['type']}")
    if "date" in doc: metadata.append(f"Date: {doc['date']}")
    if metadata: st.text("  |  ".join(metadata))

    snippet = highlight_terms(doc.get("text", "")[:500], highlight_terms_list)
    st.markdown(snippet)

    full_text = highlight_terms(doc.get("text", ""), highlight_terms_list)
    with st.expander("Show Full Text", expanded=False):
        st.markdown(full_text)

    key = f"{split}_{query_id}_{doc_id}"
    st.session_state.annotations.setdefault(key, "Unsure")
    st.session_state.annotations[key] = st.radio(
        "Annotation",
        options,
        index=options.index(st.session_state.annotations[key]),
        key=f"radio_{key}"
    )

# ======================
# QUERY NAVIGATION
# ======================
col_prev_q, col_next_q, col_save = st.sidebar.columns([1,1,2])
with col_prev_q:
    if st.button("⬅️ Previous Query"):
        st.session_state.query_idx = max(0, st.session_state.query_idx - 1)
        st.session_state.batch_start = 0
with col_next_q:
    if st.button("Next Query ➡️"):
        st.session_state.query_idx = min(len(queries) - 1, st.session_state.query_idx + 1)
        st.session_state.batch_start = 0


# ======================
# SAVE ANNOTATIONS
# ======================
def save_annotations(dataset: dict, split: str):
    relevance_map = {"Highly Relevant": 2, "Relevant": 1, "Not Relevant": 0, "Unsure": None}
    for qrel in dataset["qrels"]:
        key = f"{split}_{qrel['query_id']}_{qrel['doc_id']}"
        qrel['relevance'] = relevance_map.get(st.session_state.annotations.get(key, "Unsure"))

    # Save annotated dataset
    annotated_path = DATASET_DIR / f"qrel_{split}_annotated.json"
    annotated_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    # Save session annotations
    ANNOTATIONS_JSON.write_text(json.dumps(st.session_state.annotations, ensure_ascii=False, indent=2), encoding="utf-8")

    # Show summary
    cumulative_counts = {k:0 for k in options[:-1]}  # ignore "Unsure"
    for val in st.session_state.annotations.values():
        if val in cumulative_counts: cumulative_counts[val] += 1
    remaining = sum(1 for v in st.session_state.annotations.values() if v=="Unsure")
    batch_counts = {opt:0 for opt in options}
    for doc_id in batch_doc_ids:
        key = f"{split}_{query_id}_{doc_id}"
        batch_counts[st.session_state.annotations[key]] += 1

    st.sidebar.success(f"Saved {annotated_path} and {ANNOTATIONS_JSON}")
    st.sidebar.write("Cumulative counts (annotated only):", cumulative_counts)
    st.sidebar.write("Remaining unannotated:", remaining)
    st.sidebar.write("Current batch counts:", batch_counts)


with col_save:
    if st.button("💾 Save Annotations"):
        save_annotations(dataset, split)

