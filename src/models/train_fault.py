from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 1. Resolve project paths (SIH/)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "ch_batterygen.parquet"
MODEL_SAVE_PATH = ROOT_DIR / "saved_models" / "fault_lgb.pkl"
ENCODER_SAVE_PATH = ROOT_DIR / "saved_models" / "fault_label_encoder.pkl"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"[1/5] Loading telemetry dataset: {DATA_PATH.name}")
    df = pd.read_parquet(DATA_PATH)

    features = [
        "voltage",
        "current",
        "temperature",
        "SOC",
        "cell_voltage_min",
        "cell_voltage_max",
    ]

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=features)
    print(f"      Loaded {len(df):,} telemetry records.")

    # 2. Synthesize Physical BMS Fault Rules
    print("[2/5] Engineering BMS fault rules and balancing classes...")
    cell_delta = df["cell_voltage_max"] - df["cell_voltage_min"]

    conditions = [
        (df["temperature"] > 50.0),
        (df["cell_voltage_max"] > 4.25),
        (df["cell_voltage_min"] < 2.50),
        (cell_delta > 0.15),
    ]
    labels = ["over_temperature", "over_voltage", "under_voltage", "cell_imbalance"]

    df["fault_label_engineered"] = np.select(conditions, labels, default="normal")

    # Inject sufficient synthetic minority samples (0.5% per fault class)
    n_faults = int(len(df) * 0.005)

    idx_temp = df.sample(n_faults, random_state=42).index
    df.loc[idx_temp, "temperature"] += 35.0
    df.loc[idx_temp, "fault_label_engineered"] = "over_temperature"

    idx_volt = df.sample(n_faults, random_state=101).index
    df.loc[idx_volt, "cell_voltage_max"] += 0.85
    df.loc[idx_volt, "fault_label_engineered"] = "over_voltage"

    idx_imb = df.sample(n_faults, random_state=202).index
    df.loc[idx_imb, "cell_voltage_max"] += 0.40
    df.loc[idx_imb, "fault_label_engineered"] = "cell_imbalance"

    print("\nClass Distribution:")
    print(df["fault_label_engineered"].value_counts())

    # 3. Train / Test Split
    X = df[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["fault_label_engineered"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Balanced Classifier
    print("\n[3/5] Fitting Cost-Sensitive LightGBM Fault Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",  # Forces model to penalize minority fault errors
        min_child_samples=10,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Evaluation & Export
    print("[4/5] Evaluating performance...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("         FAULT CLASSIFICATION EVALUATION REPORT")
    print("=" * 55)
    print(f" Classification Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("=" * 55 + "\n")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    print(f"[5/5] Saved model artifact to: {MODEL_SAVE_PATH}")
    print(f"      Saved label encoder to: {ENCODER_SAVE_PATH}")


if __name__ == "__main__":
    main()