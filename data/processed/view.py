from pathlib import Path
import pandas as pd

# Target batterylife.parquet in the script's directory
script_dir = Path(__file__).parent
file_path = script_dir / "ev20.parquet"

# Configure Pandas to display ALL rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Read and print the full DataFrame
df = pd.read_parquet(file_path)
print(df)