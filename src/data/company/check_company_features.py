import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "features"
    / "company_financial_features.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("COMPANY FEATURE DATA QUALITY CHECK")
print("=" * 60)

df = pd.read_csv(FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nShape:")
print(df.shape)

print("\nCompanies:")
print(df["Company"].unique())

print("\nCompany counts:")
print(df["Company"].value_counts())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isna().sum()

print(
    missing[missing > 0]
)


# ============================================================
# INFINITE VALUES
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

infinite_count = np.isinf(
    numeric_df.to_numpy()
).sum()

print("\nInfinite numeric values:")
print(infinite_count)


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\nDuplicate Company-Year combinations:")

duplicates = df.duplicated(
    subset=["Company", "Year"]
).sum()

print(duplicates)


# ============================================================
# YEAR RANGE
# ============================================================

print("\nYear range by company:")

print(
    df.groupby("Company")["Year"]
    .agg(["min", "max", "count"])
)


# ============================================================
# TARGET CHECK
# ============================================================

print("\nTarget statistics:")

print(
    df["Next_Year_Revenue_Growth"].describe()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)