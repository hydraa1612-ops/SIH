from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Resolve project paths (SIH/)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

MODEL_PATH = ROOT_DIR / "saved_models" / "soh_lgb.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "ev20.parquet"

# Features used during training
FEATURES = [
    "voltage",
    "current",
    "temperature",
    "SOC",
    "cell_voltage_min",
    "cell_voltage_max",
]
TARGET = "capacity_Ah"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATA_PATH}")

    # 1. Load trained model artifact
    print(f"[1/4] Loading model binary from: {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)

    # 2. Load dataset
    print(f"[2/4] Loading test data from: {DATA_PATH.name}")
    df = pd.read_parquet(DATA_PATH)

    # Clean numeric data types
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])

    # Sample a test subset (100,000 random rows) for quick validation
    sample_size = min(100000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    X = sample_df[FEATURES]
    y_actual = sample_df[TARGET]

    # 3. Predict capacity
    print(f"[3/4] Running inference on {sample_size:,} sample rows...")
    y_pred = model.predict(X)

    # 4. Calculate performance metrics
    print("[4/4] Generating accuracy report...")
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    accuracy_percentage = 100 - mape

    print("\n" + "=" * 50)
    print("           PREDICTION ACCURACY REPORT")
    print("=" * 50)
    print(f" Model Accuracy Rate      : {accuracy_percentage:.2f}%")
    print(f" R² Generalization Score   : {r2:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f} Ah")
    print(f" Root Mean Sq. Error(RMSE): {rmse:.4f} Ah")
    print("=" * 50 + "\n")

    # Display side-by-side comparison table
    results_df = pd.DataFrame(
        {
            "Actual_Capacity (Ah)": y_actual.values[:10],
            "Predicted_Capacity (Ah)": np.round(y_pred[:10], 4),
            "Absolute_Error (Ah)": np.round(np.abs(y_actual.values[:10] - y_pred[:10]), 4),
        }
    )
    print("Sample Output Comparison (First 10 Rows):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()