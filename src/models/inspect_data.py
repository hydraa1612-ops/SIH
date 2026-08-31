from pathlib import Path
import pandas as pd

# Path(__file__) = SIH/src/models/inspect_data.py
# .parents[2]    = SIH/ (project root)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "ch_batterygen.parquet"

if not DATA_PATH.exists():
    print(f"File not found at: {DATA_PATH}")
    print("Files currently in data/processed/:")
    processed_dir = ROOT_DIR / "data" / "processed"
    if processed_dir.exists():
        for f in processed_dir.iterdir():
            print(f" - {f.name}")
    else:
        print("  data/processed folder does not exist.")
else:
    df = pd.read_parquet(DATA_PATH)
    print("=== DATASET SUMMARY ===")
    print(f"Total rows: {len(df):,}")
    print("\nNon-Null Counts & Data Types:")
    print(df.info())
    print("\nFirst 2 rows sample:")
    print(df.head(2).T)