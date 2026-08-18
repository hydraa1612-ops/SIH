import os
import glob
import pandas as pd

RAW_DIR = "data/raw/nasa"
OUTPUT_FILE = "data/processed/nasa.parquet"

def process_nasa():
    all_dfs = []
    # Look for both CSV exports or MAT parsed files
    files = glob.glob(os.path.join(RAW_DIR, "**/*.csv"), recursive=True)
    
    for filepath in files:
        cell_id = os.path.splitext(os.path.basename(filepath))[0]
        raw = pd.read_csv(filepath)
        
        df = pd.DataFrame({
            "source_id": "nasa",
            "battery_id": cell_id,
            "chemistry": "LCO",  # NASA standard 18650 chemistry
            "pack_or_cell": "cell",
            "timestamp": raw.get("time", raw.get("timestamp", None)),
            "voltage": raw.get("voltage_measured", raw.get("voltage", None)),
            "current": raw.get("current_measured", raw.get("current", None)),
            "temperature": raw.get("temperature_measured", raw.get("temperature", None)),
            "SOC": None,
            "cell_voltage_min": None,
            "cell_voltage_max": None,
            "capacity_Ah": raw.get("capacity", None),
            "cycle_index": raw.get("cycle", None),
            "operating_state": raw.get("type", "unknown"),
            "fault_label": "normal",
            "fault_severity": None,
            "EOL_definition": "70% of rated capacity",
            "split_group": "train"
        })
        all_dfs.append(df)
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved NASA dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_nasa()