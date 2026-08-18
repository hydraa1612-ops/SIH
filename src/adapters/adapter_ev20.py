import os
import glob
import pandas as pd

RAW_DIR = "data/raw/ev20"
OUTPUT_FILE = "data/processed/ev20.parquet"

def process_ev20():
    all_dfs = []
    # Find all CSV files in the ev20 directory
    csv_files = glob.glob(os.path.join(RAW_DIR, "**/*.csv"), recursive=True)
    
    for filepath in csv_files:
        vehicle_id = os.path.splitext(os.path.basename(filepath))[0]
        raw = pd.read_csv(filepath)
        
        df = pd.DataFrame({
            "source_id": "ev20",
            "battery_id": vehicle_id,
            "chemistry": "NCM",
            "pack_or_cell": "pack",
            "timestamp": pd.to_datetime(raw.get("time", raw.get("timestamp", None))),
            "voltage": raw.get("voltage", raw.get("V", None)),
            "current": raw.get("current", raw.get("I", None)),
            "temperature": raw.get("temperature", raw.get("T", None)),
            "SOC": raw.get("SOC", None),
            "cell_voltage_min": raw.get("cell_voltage_min", None),
            "cell_voltage_max": raw.get("cell_voltage_max", None),
            "capacity_Ah": raw.get("capacity", None),
            "cycle_index": None,
            "operating_state": "charging",
            "fault_label": "normal",
            "fault_severity": None,
            "EOL_definition": "80% of rated capacity",
            "split_group": "train"
        })
        all_dfs.append(df)
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved EV20 dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_ev20()