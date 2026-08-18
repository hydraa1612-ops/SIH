import os
import glob
import pandas as pd

RAW_DIR = "data/raw/ch_batterygen"
OUTPUT_FILE = "data/processed/ch_batterygen.parquet"

def process_ch_batterygen():
    all_dfs = []
    files = glob.glob(os.path.join(RAW_DIR, "**/*.csv"), recursive=True)
    
    for filepath in files:
        sample_id = os.path.splitext(os.path.basename(filepath))[0]
        raw = pd.read_csv(filepath)
        
        df = pd.DataFrame({
            "source_id": "ch_batterygen",
            "battery_id": raw.get("battery_id", sample_id),
            "chemistry": raw.get("chemistry", "LFP"),
            "pack_or_cell": "pack",
            "timestamp": pd.to_datetime(raw.get("timestamp", None)),
            "voltage": raw.get("voltage", None),
            "current": raw.get("current", None),
            "temperature": raw.get("temperature", None),
            "SOC": raw.get("SOC", None),
            "cell_voltage_min": raw.get("cell_voltage_min", None),
            "cell_voltage_max": raw.get("cell_voltage_max", None),
            "capacity_Ah": raw.get("capacity_Ah", None),
            "cycle_index": raw.get("cycle_index", None),
            "operating_state": raw.get("operating_state", "unknown"),
            "fault_label": raw.get("fault_label", "normal"),
            "fault_severity": raw.get("fault_severity", None),
            "EOL_definition": "80% of rated capacity",
            "split_group": "train"
        })
        all_dfs.append(df)
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved CH_BatteryGen dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_ch_batterygen()