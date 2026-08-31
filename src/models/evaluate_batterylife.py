from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "batterylife.parquet"
SOH_MODEL_PATH = ROOT_DIR / "saved_models" / "soh_lgb.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"[1/2] Loading cross-check dataset: {DATA_PATH.name}")
    df = pd.read_parquet(DATA_PATH)
    print(f"      Loaded {len(df):,} total rows across diverse cell chemistries.")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe mapping of cell voltage channels with fallbacks
    base_v = df.get("voltage", pd.Series(np.nan, index=df.index))
    df["cell_voltage_min"] = df.get("cell_voltage_min", base_v).fillna(base_v).fillna(3.6)
    df["cell_voltage_max"] = df.get("cell_voltage_max", base_v).fillna(base_v).fillna(3.6)
    df["cell_voltage_avg"] = (df["cell_voltage_min"] + df["cell_voltage_max"]) / 2.0

    df["temperature"] = df.get("temperature", pd.Series(25.0, index=df.index)).fillna(25.0)
    df["SOC"] = df.get("SOC", pd.Series(50.0, index=df.index)).fillna(50.0)

    features = ["cell_voltage_avg", "cell_voltage_min", "cell_voltage_max", "temperature", "SOC"]

    # Detect capacity or SOH target column variation
    cap_col = next(
        (c for c in ["capacity_Ah", "capacity", "cap_Ah", "SOH"] if c in df.columns and df[c].notna().sum() > 0),
        None,
    )

    if cap_col:
        df["target_cap"] = df[cap_col]
    else:
        print("      Notice: No explicit capacity column found. Deriving capacity proxy for evaluation...")
        df["target_cap"] = 1.0 - (np.abs(df["cell_voltage_max"] - df["cell_voltage_min"]) * 0.5)

    valid_df = df.dropna(subset=features + ["target_cap"])

    if len(valid_df) == 0:
        print("      Error: No valid telemetry rows remaining after preprocessing.")
        return

    sample_size = min(50000, len(valid_df))
    sample_df = valid_df.sample(n=sample_size, random_state=42)
    X = sample_df[features]

    # Calculate actual SOH percentage
    actual_cap = sample_df["target_cap"]
    nominal_cap = actual_cap.max() if actual_cap.max() > 0 else 1.0
    actual_soh_pct = (actual_cap / nominal_cap) * 100.0 if cap_col != "SOH" else actual_cap

    soh_model = joblib.load(SOH_MODEL_PATH)
    pred_soh_pct = soh_model.predict(X)

    mae = mean_absolute_error(actual_soh_pct, pred_soh_pct)
    r2 = r2_score(actual_soh_pct, pred_soh_pct)

    print("\n" + "=" * 50)
    print("      CELL-LEVEL FEATURE SOH CROSS-CHECK")
    print("=" * 50)
    print(f" Target Column Used       : {cap_col if cap_col else 'Proxy (Derived)'}")
    print(f" Sample Nominal Capacity : {nominal_cap:.2f} Ah")
    print(f" Cross-Check MAE         : {mae:.2f}%")
    print(f" Cross-Check R²          : {r2:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()