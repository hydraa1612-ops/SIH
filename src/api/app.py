from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# 1. Resolve Project Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"

SOH_MODEL_PATH = SAVED_MODELS_DIR / "soh_lgb.pkl"
FAULT_MODEL_PATH = SAVED_MODELS_DIR / "fault_lgb.pkl"
ENCODER_PATH = SAVED_MODELS_DIR / "fault_label_encoder.pkl"
RUL_MODEL_PATH = SAVED_MODELS_DIR / "rul_lgb.pkl"

app = FastAPI(
    title="EV Battery Management System (BMS) Diagnostics API",
    description="Real-time ML inference for SOH%, Fault Detection, and Remaining Useful Life (RUL).",
    version="1.0.0",
)

# Global model containers
models = {}


class TelemetryInput(BaseModel):
    voltage: float = Field(..., example=115.2, description="Pack total voltage (V)")
    current: float = Field(..., example=-12.5, description="Pack current (A)")
    temperature: float = Field(..., example=28.5, description="Pack temperature (°C)")
    SOC: float = Field(..., example=82.0, description="State of Charge (%)")
    cell_voltage_min: float = Field(..., example=3.58, description="Minimum cell voltage (V)")
    cell_voltage_max: float = Field(..., example=3.62, description="Maximum cell voltage (V)")


@app.on_event("startup")
def load_models():
    """Load joblib artifacts into memory at server startup."""
    try:
        models["soh"] = joblib.load(SOH_MODEL_PATH)
        models["fault"] = joblib.load(FAULT_MODEL_PATH)
        models["fault_encoder"] = joblib.load(ENCODER_PATH)
        models["rul"] = joblib.load(RUL_MODEL_PATH)
        print(" -> All ML model artifacts successfully loaded into memory.")
    except Exception as e:
        print(f" -> Failed to load model artifacts: {e}")


@app.get("/health", tags=["Status"])
def health_check():
    """Health check endpoint to confirm API operational status."""
    return {
        "status": "online",
        "models_loaded": len(models) == 4,
    }


@app.post("/predict", tags=["Inference"])
def predict_bms_diagnostics(telemetry: TelemetryInput):
    """Run unified inference pipeline across SOH%, Faults, and RUL."""
    if len(models) < 4:
        raise HTTPException(status_code=500, detail="Model artifacts not fully loaded.")

    try:
        data = telemetry.dict()
        
        # Derive cell-level voltage average
        cell_v_avg = (data["cell_voltage_min"] + data["cell_voltage_max"]) / 2.0

        # DataFrame inputs matching model feature expectations
        cell_features_df = pd.DataFrame([{
            "cell_voltage_avg": cell_v_avg,
            "cell_voltage_min": data["cell_voltage_min"],
            "cell_voltage_max": data["cell_voltage_max"],
            "temperature": data["temperature"],
            "SOC": data["SOC"],
        }])

        pack_features_df = pd.DataFrame([{
            "voltage": data["voltage"],
            "current": data["current"],
            "temperature": data["temperature"],
            "SOC": data["SOC"],
            "cell_voltage_min": data["cell_voltage_min"],
            "cell_voltage_max": data["cell_voltage_max"],
        }])

        # 1. Predict SOH%
        soh_pct = float(models["soh"].predict(cell_features_df)[0])
        soh_pct = float(np.clip(soh_pct, 0.0, 100.0))

        # 2. Classify Fault Status
        fault_code = models["fault"].predict(pack_features_df)[0]
        fault_label = str(models["fault_encoder"].inverse_transform([fault_code])[0])

        # 3. Predict RUL Cycles
        rul_cycles = max(0.0, float(models["rul"].predict(pack_features_df)[0]))

        return {
            "soh_percentage": round(soh_pct, 2),
            "fault_status": fault_label,
            "estimated_rul_cycles": int(round(rul_cycles)),
            "telemetry_received": data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")