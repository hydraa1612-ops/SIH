import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/ev20")
OUTPUT_FILE = Path("data/processed/ev20.parquet")

def process_ev20():
    all_dfs = []
    
    csv_files = [f for f in RAW_DIR.rglob("*") if f.suffix.lower() == ".csv"]
    print(f"Found {len(csv_files)} CSV files")
    
    for filepath in csv_files:
        vehicle_id = filepath.stem
        try:
            raw = pd.read_csv(filepath)
            
            # Map column names (lowercase & stripped) to actual original header names
            lower_to_orig = {str(col).strip().lower(): col for col in raw.columns}
            
            # Substring helper: finds the first header containing any key pattern
            def get_col(patterns):
                for pattern in patterns:
                    for low_col, orig_col in lower_to_orig.items():
                        if pattern in low_col:
                            return raw[orig_col]
                return None

            # Optional: Debug line to print raw columns on the first file
            if filepath == csv_files[0]:
                print(f"Sample file raw columns: {list(raw.columns)}")

            df = pd.DataFrame({
                "source_id": "ev20",
                "battery_id": vehicle_id,
                "chemistry": "NCM",
                "pack_or_cell": "pack",
                "timestamp": pd.to_datetime(get_col(["time", "date"]), errors="coerce"),
                "voltage": get_col(["volt", "v_"]),
                "current": get_col(["curr", "i_"]),
                "temperature": get_col(["temp", "t_"]),
                "SOC": get_col(["soc"]),
                "cell_voltage_min": get_col(["min_cell", "cell_voltage_min"]),
                "cell_voltage_max": get_col(["max_cell", "cell_voltage_max"]),
                "capacity_Ah": get_col(["cap", "ah"]),
                "cycle_index": None,
                "operating_state": "charging",
                "fault_label": "normal",
                "fault_severity": None,
                "EOL_definition": "80% of rated capacity",
                "split_group": "train"
            })
            
            all_dfs.append(df)
            print(f"Loaded {filepath.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipping {filepath.name}: {e}")
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Deduplicate exact duplicate rows if present
        initial_len = len(final_df)
        final_df = final_df.drop_duplicates()
        print(f"Removed {initial_len - len(final_df)} duplicate rows.")
        
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved EV20 dataset ({len(final_df)} total rows) to {OUTPUT_FILE}")
    else:
        print("x No readable CSV files found.")

if __name__ == "__main__":
    process_ev20()