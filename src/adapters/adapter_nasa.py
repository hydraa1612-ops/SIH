from pathlib import Path
import pandas as pd
import scipy.io as sio

RAW_DIR = Path("data/raw/nasa")
OUTPUT_FILE = Path("data/processed/nasa.parquet")

CANONICAL_COLUMNS = [
    "source_id", "battery_id", "chemistry", "pack_or_cell", 
    "timestamp", "voltage", "current", "temperature", "SOC", 
    "cell_voltage_min", "cell_voltage_max", "capacity_Ah", 
    "cycle_index", "operating_state", "fault_label", 
    "fault_severity", "EOL_definition", "split_group"
]

def process_nasa():
    print(f"Scanning NASA raw directory: {RAW_DIR.resolve()}")
    
    mat_files = list(RAW_DIR.rglob("*.mat"))
    csv_files = list(RAW_DIR.rglob("*.csv"))
    
    print(f"Found {len(mat_files)} MAT files and {len(csv_files)} CSV files.")
    
    all_dfs = []

    # Process CSV files if present
    for filepath in csv_files:
        try:
            raw = pd.read_csv(filepath)
            if raw.empty:
                continue
            
            cols = {str(c).strip().lower(): c for c in raw.columns}
            def get_col(candidates):
                for c in candidates:
                    if c in cols:
                        return raw[cols[c]]
                return None

            df = pd.DataFrame({
                "source_id": "nasa",
                "battery_id": filepath.stem,
                "chemistry": "LiCoO2",
                "pack_or_cell": "cell",
                "timestamp": pd.to_datetime(get_col(["timestamp", "time"]), errors="coerce"),
                "voltage": get_col(["voltage", "v", "voltage_measured"]),
                "current": get_col(["current", "i", "current_measured"]),
                "temperature": get_col(["temperature", "temp", "temperature_measured"]),
                "SOC": get_col(["soc"]),
                "cell_voltage_min": None,
                "cell_voltage_max": None,
                "capacity_Ah": get_col(["capacity", "capacity_ah"]),
                "cycle_index": get_col(["cycle", "cycle_index"]),
                "operating_state": "unknown",
                "fault_label": "normal",
                "fault_severity": None,
                "EOL_definition": "70% of initial capacity (1.4 Ah)",
                "split_group": "train"
            })
            all_dfs.append(df)
            print(f"Loaded CSV: {filepath.name}")
        except Exception as e:
            print(f"Skipping CSV {filepath.name}: {e}")

    # Process MATLAB .mat files
    for filepath in mat_files:
        try:
            mat = sio.loadmat(filepath)
            bat_name = filepath.stem
            
            if bat_name in mat:
                data = mat[bat_name]
                cycles = data[0, 0]["cycle"][0]
                
                records = []
                for idx, cycle in enumerate(cycles):
                    cycle_type = str(cycle["type"][0])
                    cycle_data = cycle["data"]
                    
                    try:
                        v = cycle_data["Voltage_measured"][0, 0].flatten()
                        i = cycle_data["Current_measured"][0, 0].flatten()
                        t = cycle_data["Temperature_measured"][0, 0].flatten()
                        
                        time_sec = cycle_data["Time"][0, 0].flatten() if "Time" in cycle_data.dtype.names else None
                        
                        cap = None
                        if "Capacity" in cycle_data.dtype.names and len(cycle_data["Capacity"][0, 0]) > 0:
                            cap = cycle_data["Capacity"][0, 0][0, 0]

                        for step in range(len(v)):
                            records.append({
                                "source_id": "nasa",
                                "battery_id": bat_name,
                                "chemistry": "LiCoO2",
                                "pack_or_cell": "cell",
                                "timestamp": time_sec[step] if time_sec is not None else None,
                                "voltage": v[step],
                                "current": i[step],
                                "temperature": t[step],
                                "SOC": None,
                                "cell_voltage_min": None,
                                "cell_voltage_max": None,
                                "capacity_Ah": cap,
                                "cycle_index": idx + 1,
                                "operating_state": cycle_type,
                                "fault_label": "normal",
                                "fault_severity": None,
                                "EOL_definition": "70% of initial capacity (1.4 Ah)",
                                "split_group": "train"
                            })
                    except Exception:
                        continue

                if records:
                    df = pd.DataFrame(records)
                    all_dfs.append(df)
                    print(f"Loaded MAT: {filepath.name} ({len(records)} measurements)")
        except Exception as e:
            print(f"Skipping MAT {filepath.name}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"✓ Saved NASA dataset to {OUTPUT_FILE}")
    else:
        empty_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        empty_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"⚠ No readable MAT/CSV files parsed in NASA directory. Created stub parquet at {OUTPUT_FILE}")

if __name__ == "__main__":
    process_nasa()