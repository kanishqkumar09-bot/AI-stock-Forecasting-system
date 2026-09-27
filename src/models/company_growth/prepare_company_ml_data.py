import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "features"
    / "company_financial_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("COMPANY GROWTH ML DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# SORT DATA
# ============================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df = df.sort_values(
    ["Company", "Year"]
).reset_index(drop=True)


# ============================================================
# SELECT FEATURES
# ============================================================

features = [
    "Revenue_Growth",
    "Profit_Growth",
    "EPS_Growth",
    "EBIT_Margin",
    "Net_Profit_Margin",
    "Revenue_Growth_Change",
    "Profit_Growth_Change",
    "EPS_Growth_Change"
]

target = "Next_Year_Revenue_Growth"


# ============================================================
# CHECK AVAILABLE COLUMNS
# ============================================================

missing_columns = [
    col for col in features + [target]
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    raise SystemExit


# ============================================================
# CREATE ML DATASET
# ============================================================

ml_df = df[
    ["Company", "Year"] + features + [target]
].copy()


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

before = len(ml_df)

ml_df = ml_df.dropna(
    subset=[target]
).copy()

ml_df = ml_df.dropna(
    subset=features
).copy()


after = len(ml_df)

print("\nTarget cleaning:")
print(f"Rows before: {before}")
print(f"Rows after : {after}")
print(f"Rows removed: {before - after}")


# ============================================================
# CHECK MISSING FEATURE VALUES
# ============================================================

print("\nMissing feature values:")

missing = ml_df[features].isna().sum()

print(
    missing[missing > 0]
)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

numeric_columns = features + [target]

infinite_rows = np.isinf(
    ml_df[numeric_columns]
    .select_dtypes(include=np.number)
).any(axis=1).sum()

print("\nRows containing infinite values:")
print(infinite_rows)


# ============================================================
# SAVE ML DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "company_growth_ml_dataset.csv"
)

ml_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ML DATASET SUMMARY")
print("=" * 60)

print(f"\nRows: {len(ml_df)}")
print(f"Features: {len(features)}")

print("\nCompanies:")
print(
    ml_df["Company"]
    .value_counts()
)

print("\nFeature columns:")
for feature in features:
    print(f" - {feature}")

print(f"\nTarget:")
print(f" - {target}")

print("\nDataset saved to:")
print(output_file)

print("\n" + "=" * 60)
print("COMPANY ML DATA PREPARATION COMPLETE")
print("=" * 60)