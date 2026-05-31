# scripts/build_data_pipeline.py
import sys
from pathlib import Path

# -----------------------------
# Add project root to sys.path
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# -----------------------------
# Imports
# -----------------------------
from src.data_building.build_corpus import process_pdfs
from src.data_building.generate_query_pool import GoldQueryPool
from src.data_building.split_qrels import build_qrels_and_split


# -----------------------------
# Pipeline
# -----------------------------
def main():
    print("🚀 Starting data building pipeline...")

    # Step 1: Process PDFs into corpus
    print("\n📄 Step 1: Processing PDFs into corpus...")
    process_pdfs(
        raw_pdf_dir=project_root / "data/raw/pdfs",
        raw_text_dir=project_root / "data/raw/text",
        processed_dir=project_root / "data/processed",
        interim_dir=project_root / "data/interim",
        corpus_file=project_root / "data/interim/corpus.json",
        corpus_preprocessed_file=project_root / "data/processed/corpus_preprocessed.json",
        metadata_file=project_root / "data/interim/metadata.csv",
        save_raw_text=True
    )
    print("✅ PDFs processed and corpus saved.")

    # Step 2: Build gold query pool
    print("\n🔍 Step 2: Generating gold query pool...")
    corpus_path = project_root / "data/processed/corpus_preprocessed.json"
    queries_path = project_root / "data/processed/queries.json"

    gqp = GoldQueryPool(
        corpus_path=corpus_path,
        queries_path=queries_path,
        max_pool_size=100
    )
    gqp.build_pool()
    gqp.save(project_root / "data/processed/gold_query_pool.json")
    print("✅ Gold query pool built and saved.")

    # Step 3: Split Qrels into train/test
    print("\n📊 Step 3: Splitting Qrels into train/test sets...")
    build_qrels_and_split(
        input_file=project_root / "data/processed/gold_query_pool.json",
        corpus_meta_file=project_root / "data/processed/corpus_preprocessed.json",
        train_file=project_root / "data/processed/qrels/qrel_train.json",
        test_file=project_root / "data/processed/qrels/qrel_test.json",
        train_ratio=0.8,
        random_seed=42
    )
    print("✅ Qrels split completed.")

    print("\n🎉 Data building pipeline finished successfully!")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
