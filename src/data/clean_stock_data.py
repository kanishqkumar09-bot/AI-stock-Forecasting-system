import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "stock" / "TCS.csv"

PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed" / "stock"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_FOLDER / "TCS.csv"


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 60)
print("LOADING RAW TCS DATA")
print("=" * 60)

df = pd.read_csv(RAW_FILE)


# ============================================================
# DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# SORT BY DATE
# ============================================================

df = df.sort_values("Date")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["Date"])

after = len(df)

print(f"\nDuplicate rows removed: {before - after}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
)

after = len(df)

print(f"Invalid rows removed: {before - after}")


# ============================================================
# PRICE LOGIC CHECK
# ============================================================

invalid_prices = (
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"])
)

print(f"Invalid price rows: {invalid_prices.sum()}")

df = df[~invalid_prices]


# ============================================================
# VOLUME CHECK
# ============================================================

invalid_volume = df["Volume"] < 0

print(f"Invalid volume rows: {invalid_volume.sum()}")

df = df[~invalid_volume]


# ============================================================
# FINAL SORT
# ============================================================

df = df.sort_values("Date")

df = df.reset_index(drop=True)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(f"\nFinal rows: {len(df)}")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nDate range:")
print(
    df["Date"].min(),
    "→",
    df["Date"].max()
)

print("\nSaved processed dataset:")
print(OUTPUT_FILE)