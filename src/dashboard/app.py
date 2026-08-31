from pathlib import Path
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="EV BMS Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Real-Time EV Battery Analytics Dashboard")
st.markdown("Live SOH Estimation, Fault Detection, and RUL Diagnostics")

# Sidebar - Live Telemetry Controls
st.sidebar.header("🕹️ Live Telemetry Inputs")

voltage = st.sidebar.slider("Pack Voltage (V)", 80.0, 150.0, 115.2, 0.1)
current = st.sidebar.slider("Pack Current (A)", -100.0, 100.0, -12.5, 0.1)
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 80.0, 28.5, 0.1)
soc = st.sidebar.slider("State of Charge (%)", 0.0, 100.0, 82.0, 1.0)
cell_v_min = st.sidebar.slider("Min Cell Voltage (V)", 2.5, 4.5, 3.58, 0.01)
cell_v_max = st.sidebar.slider("Max Cell Voltage (V)", 2.5, 4.5, 3.62, 0.01)

payload = {
    "voltage": voltage,
    "current": current,
    "temperature": temperature,
    "SOC": soc,
    "cell_voltage_min": cell_v_min,
    "cell_voltage_max": cell_v_max,
}

# Query Backend API
try:
    response = requests.post(API_URL, json=payload, timeout=5)
    if response.status_code == 200:
        data = response.json()

        # Top Metric Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("State of Health (SOH)", f"{data['soh_percentage']}%")

        with col2:
            status = data["fault_status"].upper()
            if status == "NORMAL":
                st.success(f"Fault Status: {status}")
            else:
                st.error(f"Fault Status: {status}")

        with col3:
            st.metric("Estimated RUL", f"{data['estimated_rul_cycles']} Cycles")

        st.divider()

        # Gauge Chart for SOH%
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=data["soh_percentage"],
                title={"text": "Battery Health Gauge (SOH%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 70], "color": "#ff4b4b"},
                        {"range": [70, 85], "color": "#ffa500"},
                        {"range": [85, 100], "color": "#00cc96"},
                    ],
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"API Error: {response.status_code}")

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to FastAPI server. Make sure Uvicorn is running on port 8000.")