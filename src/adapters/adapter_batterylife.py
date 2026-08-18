import h5py
import os
from pathlib import Path
import numpy as np
import pandas as pd

# --- Paths ---
RAW_DIR = Path("data/raw/batterylife")
PROCESSED_FILE = Path("data/processed/batterylife.parquet")

def load_real_matr_data(mat_files):
    """Parses MATR dataset HDF5 structure across all cell references."""
    records = []
    print(f"Processing {len(mat_files)} raw MATR file(s)...")

    for file_path in mat_files:
        print(f"Reading: {file_path.name}")
        try:
            with h5py.File(file_path, 'r') as f:
                if 'batch' not in f:
                    print("--> 'batch' key not found in HDF5 file.")
                    continue

                batch = f['batch']
                summary_ds = batch['summary']  # HDF5 object references

                # Flatten array of references so we can reliably iterate over every single cell
                refs = summary_ds[()].flatten()
                num_cells = len(refs)
                print(f"Found {num_cells} unique cells in {file_path.name}.")

                for cell_idx, ref in enumerate(refs):
                    battery_id = f"matr_cell_{cell_idx + 1}"
                    
                    try:
                        # Dereference cell summary
                        cell_summary = f[ref]

                        # Extract arrays safely
                        cycles_arr = np.array(cell_summary['cycle']).flatten() if 'cycle' in cell_summary else []
                        cap_arr = np.array(cell_summary['QDischarge']).flatten() if 'QDischarge' in cell_summary else []
                        temp_arr = np.array(cell_summary['Tavg']).flatten() if 'Tavg' in cell_summary else []

                        num_entries = len(cap_arr)
                        if num_entries == 0:
                            continue

                        for i in range(num_entries):
                            records.append({
                                "source_id": "batterylife",
                                "battery_id": battery_id,
                                "chemistry": "LFP",
                                "pack_or_cell": "cell",
                                "timestamp": None,
                                "voltage": None,
                                "current": None,
                                "temperature": float(temp_arr[i]) if i < len(temp_arr) else None,
                                "SOC": None,
                                "cell_voltage_min": None,
                                "cell_voltage_max": None,
                                "capacity_Ah": float(cap_arr[i]),
                                "cycle_index": int(cycles_arr[i]) if i < len(cycles_arr) else i + 1,
                                "operating_state": "discharging",
                                "fault_label": "normal",
                                "fault_severity": None,
                                "EOL_definition": "80% capacity retention",
                                "split_group": "train"
                            })
                    except Exception as e:
                        print(f"Skipping cell {cell_idx + 1}: {e}")

            print(f"Successfully loaded {file_path.name}")

        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")

    return pd.DataFrame(records)


def process_batterylife():
    mat_files = [f for f in RAW_DIR.rglob("*") if f.suffix.lower() == ".mat" or "batch" in f.name.lower()]
    mat_files = list(set(mat_files))

    if mat_files:
        df = load_real_matr_data(mat_files)
    else:
        print("No raw files found in data/raw/batterylife.")
        df = pd.DataFrame()

    if not df.empty:
        PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_FILE, index=False)
        print(f"✓ Saved batterylife dataset ({len(df)} rows, {df['battery_id'].nunique()} cells) to {PROCESSED_FILE}")
    else:
        print("x Failed to extract data.")

if __name__ == "__main__":
    process_batterylife()