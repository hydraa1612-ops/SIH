import os
import pandas as pd

def qc_report(df, name):
    print(f"\n==========================================")
    print(f"--- QC report: {name} ---")
    print(f"==========================================")
    print("Total Rows:", len(df))
    print("Duplicate rows:", df.duplicated().sum())
    print("\nMissing values per column:\n", df.isna().sum())
    
    # Check for voltage anomalies if voltage exists
    if "voltage" in df.columns and df["voltage"].dropna().count() > 0:
        print(f"\nVoltage range: {df['voltage'].min()} V - {df['voltage'].max()} V")
        bad_voltage = df[(df["voltage"] < 1.5) | (df["voltage"] > 5.0)]
        print(f"Suspicious voltage rows (<1.5V or >5.0V): {len(bad_voltage)}")
        
    # Check temperature range
    if "temperature" in df.columns and df["temperature"].dropna().count() > 0:
        print(f"Temperature range: {df['temperature'].min()}°C - {df['temperature'].max()}°C")
        
    if "battery_id" in df.columns:
        print(f"Unique battery IDs: {df['battery_id'].nunique()}")

def run_all_qc():
    os.makedirs("data/qc_reports", exist_ok=True)
    sources = ["ev20", "batterylife", "nasa", "ch_batterygen"]
    
    for src in sources:
        path = f"data/processed/{src}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            qc_report(df, src)
            
            # Save the report text file as required by section 5.3
            report_path = f"data/qc_reports/{src}.txt"
            with open(report_path, "w") as f:
                f.write(f"--- QC Report for {src} ---\n")
                f.write(f"Total Rows: {len(df)}\n")
                f.write(f"Unique Batteries: {df['battery_id'].nunique() if 'battery_id' in df else 'N/A'}\n")
                f.write(f"Duplicate Rows: {df.duplicated().sum()}\n\n")
                f.write(f"Missing Values per Column:\n{df.isna().sum().to_string()}\n")
            print(f"--> Saved QC report to {report_path}")
        else:
            print(f"\n[SKIP] {path} not found. Run its adapter script first if needed.")

if __name__ == "__main__":
    run_all_qc()