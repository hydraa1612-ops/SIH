from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/ch_batterygen")
OUTPUT_FILE = Path("data/processed/ch_batterygen.parquet")

CANONICAL_COLUMNS = [
    "source_id", "battery_id", "chemistry", "pack_or_cell", 
    "timestamp", "voltage", "current", "temperature", "SOC", 
    "cell_voltage_min", "cell_voltage_max", "capacity_Ah", 
    "cycle_index", "operating_state", "fault_label", 
    "fault_severity", "EOL_definition", "split_group"
]

def process_ch_batterygen():
    print(f"Scanning directory: {RAW_DIR.resolve()}")
    
    # Locate all data files inside data/raw/ch_batterygen/
    all_files = [f for f in RAW_DIR.rglob("*") if f.is_file()]
    data_files = [
        f for f in all_files 
        if f.suffix.lower() in [".csv", ".parquet", ".xlsx"]
    ]
    print(f"Found {len(data_files)} raw data candidate file(s).")
    
    all_dfs = []
    for filepath in data_files:
        try:
            if filepath.suffix.lower() == ".csv":
                raw = pd.read_csv(filepath)
            elif filepath.suffix.lower() == ".parquet":
                raw = pd.read_parquet(filepath)
            elif filepath.suffix.lower() == ".xlsx":
                raw = pd.read_excel(filepath)
            else:
                continue

            if raw.empty:
                continue

            # Case-insensitive column lookup dictionary
            cols = {str(c).strip().lower(): c for c in raw.columns}
            
            def get_col(candidates):
                for c in candidates:
                    if c in cols:
                        return raw[cols[c]]
                return None

            df = pd.DataFrame({
                "source_id": "ch_batterygen",
                "battery_id": filepath.stem,
                "chemistry": "LFP",
                "pack_or_cell": "pack",
                "timestamp": pd.to_datetime(get_col(["time", "timestamp", "t"]), errors="coerce"),
                "voltage": get_col(["sum_voltage", "voltage", "v"]),
                "current": get_col(["sum_current", "current", "i"]),
                "temperature": get_col(["max_temp", "temperature", "temp"]),
                "SOC": get_col(["soc"]),
                "cell_voltage_min": get_col(["min_cell_volt", "cell_voltage_min"]),
                "cell_voltage_max": get_col(["max_cell_volt", "cell_voltage_max"]),
                "capacity_Ah": get_col(["capacity_ah", "capacity"]),
                "cycle_index": get_col(["cycle_index", "cycle"]),
                "operating_state": "discharging",
                "fault_label": "normal",
                "fault_severity": None,
                "EOL_definition": "80% of rated capacity",
                "split_group": "train"
            })
            
            # Ensure all canonical columns exist
            for col in CANONICAL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
                    
            df = df[CANONICAL_COLUMNS]
            all_dfs.append(df)
            print(f"✓ Successfully processed {filepath.name} ({len(df)} rows)")

        except Exception as e:
            print(f"⚠ Skipping {filepath.name}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved CH_BatteryGen dataset ({len(final_df)} rows) to {OUTPUT_FILE}")
    else:
        empty_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        empty_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"⚠ No readable raw files found in {RAW_DIR}. Saved canonical stub to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_ch_batterygen()