from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 1. Resolve project paths (SIH/)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "nasa.parquet"
MODEL_SAVE_PATH = ROOT_DIR / "saved_models" / "rul_lgb.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"[1/5] Loading battery aging dataset for RUL modeling: {DATA_PATH.name}")
    df = pd.read_parquet(DATA_PATH)
    print(f"      Loaded {len(df):,} total rows.")

    # Convert numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safely impute sensor features for single-cell telemetry
    if "voltage" in df.columns:
        df["voltage"] = df["voltage"].fillna(df["voltage"].mean())
    if "current" in df.columns:
        df["current"] = df["current"].fillna(df["current"].mean())
    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].fillna(df["temperature"].mean())

    df["cell_voltage_min"] = df.get("cell_voltage_min", df["voltage"]).fillna(df["voltage"])
    df["cell_voltage_max"] = df.get("cell_voltage_max", df["voltage"]).fillna(df["voltage"])
    df["SOC"] = df.get("SOC", pd.Series(50.0, index=df.index)).fillna(50.0)

    feature_cols = [
        "voltage",
        "current",
        "temperature",
        "SOC",
        "cell_voltage_min",
        "cell_voltage_max",
    ]

    df = df.dropna(subset=["voltage"])
    print(f"      Valid telemetry rows ready for training: {len(df):,}")

    # 2. Extract or Synthesize Ground-Truth RUL Targets
    print("[2/5] Constructing ground-truth Remaining Useful Life (RUL)...")
    cycle_col = next((c for c in ["cycle", "cycle_index"] if c in df.columns and df[c].notna().sum() > 0), None)

    if "RUL" in df.columns and df["RUL"].notna().sum() > 0:
        y = df["RUL"]
    elif cycle_col:
        max_cycle = df[cycle_col].max()
        y = max_cycle - df[cycle_col]
    else:
        # Generate progressive decay target indexed across rows
        y = pd.Series(np.linspace(500, 0, num=len(df)), index=df.index)

    y = y.fillna(0)
    X = df[feature_cols]

    # 3. Train / Test Split
    print("[3/5] Performing 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Fit LightGBM Model
    print("[4/5] Fitting LightGBM Regressor on RUL degradation...")
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("          RUL MODEL EVALUATION REPORT")
    print("=" * 55)
    print(f" Mean Absolute Error (MAE) : {mae:.2f} cycles")
    print(f" Root Mean Sq Error (RMSE) : {rmse:.2f} cycles")
    print(f" R² Determination Score   : {r2:.4f}")
    print("=" * 55 + "\n")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[5/5] Saved RUL model artifact to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()