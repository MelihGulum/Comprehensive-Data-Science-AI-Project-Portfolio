# file: save_ir.py
import pickle
import faiss


def save_university_ir(ir: "UniversityIR", folder="saved_ir"):
    import os
    os.makedirs(folder, exist_ok=True)

    # # --- Save doc_json and sparse_docs ---
    # with open(f"{folder}/doc_json.pkl", "wb") as f:
    #     pickle.dump(ir.doc_json, f)
    # with open(f"{folder}/sparse_docs.pkl", "wb") as f:
    #     pickle.dump(ir.sparse_docs, f)
    # with open(f"{folder}/title_tokenized.pkl", "wb") as f:
    #     pickle.dump(ir.title_tokenized, f)

    # --- Save FAISS doc_id order ---
    with open(f"{folder}/doc_ids.pkl", "wb") as f:
        pickle.dump(ir.doc_ids, f)

    # --- Save BM25 model ---
    with open(f"{folder}/bm25.pkl", "wb") as f:
        pickle.dump(ir.bm25, f)

    # --- Save dense model (SentenceTransformer) ---
    if ir.dense_model is not None:
        ir.dense_model.save(f"{folder}/dense_model")

    # --- Save FAISS index ---
    if ir.index is not None:
        faiss.write_index(ir.index, f"{folder}/faiss.index")

    # --- Save cross-encoder model (optional) ---
    if ir.cross_encoder is not None:
        ir.cross_encoder.save(f"{folder}/cross_encoder")

    # --- Save learned weights and alpha ---
    with open(f"{folder}/weights.pkl", "wb") as f:
        pickle.dump({
            "best_alpha": ir.best_alpha,
            "alpha_by_type": ir.alpha_by_type,
            "best_weights": ir.best_weights
        }, f)

    print("IR model saved successfully.")
