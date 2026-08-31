from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

print("Checking 'fault_label' contents across all datasets:\n")

for file_path in PROCESSED_DIR.glob("*.parquet"):
    df = pd.read_parquet(file_path)
    if "fault_label" in df.columns:
        unique_vals = df["fault_label"].dropna().value_counts()
        print(f"=== {file_path.name} ===")
        print(unique_vals)
        print("-" * 40)
    else:
        print(f"=== {file_path.name} === -> No 'fault_label' column found.")