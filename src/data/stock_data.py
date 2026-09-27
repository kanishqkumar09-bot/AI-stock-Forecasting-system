import yfinance as yf  # allow financial market data from yahoo finance
import pandas as pd
from pathlib import Path  #select path where to save the data


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  #root folder

RAW_STOCK_PATH = PROJECT_ROOT / "data" / "raw" / "stock"

RAW_STOCK_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMPANY
# ============================================================

ticker = "TCS.NS"

start_date = "2016-01-01"
end_date = "2026-08-22"


# ============================================================
# DOWNLOAD HISTORICAL DATA
# ============================================================

print("=" * 60)
print("DOWNLOADING TCS STOCK DATA")
print("=" * 60)

data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    actions=True,
    progress=True
)


# ============================================================
# CHECK DATA
# ============================================================

if data.empty:
    raise ValueError("No data was downloaded. Check ticker or internet connection.")


print("\nData downloaded successfully!")

print("\nShape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())


# ============================================================
# HANDLE MULTI-INDEX COLUMNS
# ============================================================

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


# ============================================================
# RESET INDEX
# ============================================================

data.reset_index(inplace=True)


# ============================================================
# SAVE RAW DATA
# ============================================================

output_file = RAW_STOCK_PATH / "TCS.csv"

data.to_csv(output_file, index=False)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Saved to:")
print(output_file)

print(f"\nTotal rows: {len(data)}")

print("\nColumns:")
print(list(data.columns))