import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_PATH = PROJECT_ROOT / "data" / "raw" / "stock" / "TCS.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TCS DATASET VALIDATION")
print("=" * 60)

df = pd.read_csv(FILE_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# DATE CHECK
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

print("\nFirst date:")
print(df["Date"].min())

print("\nLast date:")
print(df["Date"].max())


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\nBasic statistics:")
print(df.describe())


# ============================================================
# FIRST AND LAST ROWS
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)