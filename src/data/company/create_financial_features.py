import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "raw"
    / "company_financials.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "features"
)

OUTPUT_FILE = (
    OUTPUT_FOLDER
    / "company_financial_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("COMPANY FINANCIAL FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    skipinitialspace=True
)

print("\nRaw data loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CONVERT FINANCIAL COLUMNS TO NUMERIC
# ============================================================

numeric_columns = [
    "Year",
    "Revenue",
    "EBIT",
    "Profit_Before_Tax",
    "Net_Profit",
    "EPS",
    "Net_Worth"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["Company", "Year"]
).reset_index(drop=True)


# ============================================================
# GROWTH FEATURES
# ============================================================

df["Revenue_Growth"] = (
    df.groupby("Company")["Revenue"]
    .pct_change() * 100
)

df["EBIT_Growth"] = (
    df.groupby("Company")["EBIT"]
    .pct_change() * 100
)

df["Profit_Growth"] = (
    df.groupby("Company")["Net_Profit"]
    .pct_change() * 100
)

df["EPS_Growth"] = (
    df.groupby("Company")["EPS"]
    .pct_change() * 100
)


# ============================================================
# PROFITABILITY FEATURES
# ============================================================

df["EBIT_Margin"] = np.where(
    df["Revenue"] != 0,
    (df["EBIT"] / df["Revenue"]) * 100,
    np.nan
)

df["Net_Profit_Margin"] = np.where(
    df["Revenue"] != 0,
    (df["Net_Profit"] / df["Revenue"]) * 100,
    np.nan
)


# ============================================================
# PROFITABILITY CHANGE
# ============================================================

df["EBIT_Margin_Change"] = (
    df.groupby("Company")["EBIT_Margin"]
    .diff()
)

df["Net_Profit_Margin_Change"] = (
    df.groupby("Company")["Net_Profit_Margin"]
    .diff()
)


# ============================================================
# NET WORTH GROWTH
# ============================================================

df["Net_Worth_Growth"] = (
    df.groupby("Company")["Net_Worth"]
    .pct_change() * 100
)
# ============================================================
# SCALE-NORMALIZED FINANCIAL FEATURES
# ============================================================

# Profit relative to revenue
df["Profit_to_Revenue"] = (
    df["Net_Profit"] / df["Revenue"]
) * 100

# EBIT relative to revenue
df["EBIT_to_Revenue"] = (
    df["EBIT"] / df["Revenue"]
) * 100

# Net profit relative to net worth
df["Profit_to_Net_Worth"] = np.where(
    df["Net_Worth"] != 0,
    (df["Net_Profit"] / df["Net_Worth"]) * 100,
    np.nan
)

# Revenue relative to net worth
df["Revenue_to_Net_Worth"] = np.where(
    df["Net_Worth"] != 0,
    df["Revenue"] / df["Net_Worth"],
    np.nan
)

# EPS relative to revenue scale
df["EPS_to_Revenue"] = np.where(
    df["Revenue"] != 0,
    df["EPS"] / df["Revenue"],
    np.nan
)

# ============================================================
# GROWTH MOMENTUM FEATURES
# ============================================================

df["Revenue_Growth_Change"] = (
    df.groupby("Company")["Revenue_Growth"]
    .diff()
)

df["Profit_Growth_Change"] = (
    df.groupby("Company")["Profit_Growth"]
    .diff()
)

df["EPS_Growth_Change"] = (
    df.groupby("Company")["EPS_Growth"]
    .diff()
)



# ============================================================
# TARGET
# ============================================================
# We want to predict NEXT year's revenue growth.
#
# Example:
# 2024 row → target = 2025 revenue growth
# 2025 row → target = 2026 revenue growth
#
# Therefore we shift Revenue_Growth upward.

df["Next_Year_Revenue_Growth"] = (
    df.groupby("Company")["Revenue_Growth"]
    .shift(-1)
)


# ============================================================
# DISPLAY FEATURES
# ============================================================

print("\n" + "=" * 60)
print("ENGINEERED FEATURES")
print("=" * 60)

display_columns = [
    "Company",
    "Year",
    "Revenue",
    "Net_Profit",
    "EPS",
    "Revenue_Growth",
    "Profit_Growth",
    "EPS_Growth",
    "EBIT_Margin",
    "Net_Profit_Margin",
    "EBIT_Margin_Change",
    "Net_Profit_Margin_Change",
    "Net_Worth_Growth",
    "Next_Year_Revenue_Growth"
]

print(
    df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================
# The latest year doesn't have a known next-year growth value.

before = len(df)

df_model = df.dropna(
    subset=["Next_Year_Revenue_Growth"]
).copy()

after = len(df_model)

print("\nRows before target removal:", before)
print("Rows after target removal :", after)
print("Rows removed              :", before - after)

# ============================================================
# HANDLE MISSING VALUES IN ENGINEERED FEATURES
# ============================================================

TARGET = "Next_Year_Revenue_Growth"

feature_cols = [
    col for col in df.columns
    if col not in ["Company", "Year", TARGET]
]

df[feature_cols] = df[feature_cols].fillna(0)

print("\nMissing values after feature cleaning:")
print(df.isna().sum())

print("\nTOTAL MISSING:", df.isna().sum().sum())

# ============================================================
# SAVE
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

df_model.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print("\nFinal dataset shape:")
print(df_model.shape)

print("\nSaved to:")
print(OUTPUT_FILE)