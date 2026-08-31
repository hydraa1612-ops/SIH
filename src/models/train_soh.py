from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "ev20.parquet"
FALLBACK_DATA_PATH = ROOT_DIR / "data" / "processed" / "ch_batterygen.parquet"
MODEL_SAVE_PATH = ROOT_DIR / "saved_models" / "soh_lgb.pkl"


def main():
    target_path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    print(f"[1/4] Loading dataset: {target_path.name}")
    df = pd.read_parquet(target_path)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safely derive cell-level voltage metrics
    if "cell_voltage_min" not in df.columns or df["cell_voltage_min"].isna().all():
        df["cell_voltage_min"] = df["voltage"] / 32.0  # approximate series cell count
    if "cell_voltage_max" not in df.columns or df["cell_voltage_max"].isna().all():
        df["cell_voltage_max"] = df["voltage"] / 32.0

    df["cell_voltage_avg"] = (df["cell_voltage_min"] + df["cell_voltage_max"]) / 2.0
    df["temperature"] = df.get("temperature", pd.Series(25.0, index=df.index)).fillna(25.0)
    df["SOC"] = df.get("SOC", pd.Series(50.0, index=df.index)).fillna(50.0)

    # Scale-invariant feature set (excludes raw pack voltage)
    features = ["cell_voltage_avg", "cell_voltage_min", "cell_voltage_max", "temperature", "SOC"]

    # Target SOH%
    nominal_cap = df["capacity_Ah"].max() if "capacity_Ah" in df.columns else 120.0
    df["SOH_target"] = (df["capacity_Ah"] / nominal_cap) * 100.0 if "capacity_Ah" in df.columns else df["SOH"]

    df = df.dropna(subset=features + ["SOH_target"])
    X = df[features]
    y = df["SOH_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[2/4] Fitting Scale-Invariant LightGBM Model...")
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, verbosity=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[3/4] EV Pack Test Set MAE: {mean_absolute_error(y_test, y_pred):.2f}% | R²: {r2_score(y_test, y_pred):.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[4/4] Saved model to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()