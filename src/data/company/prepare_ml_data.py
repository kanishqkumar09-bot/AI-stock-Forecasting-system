import sys
from pathlib import Path

# ==============================
# PROJECT PATH
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ==============================
# IMPORTS
# ==============================

import pandas as pd
import numpy as np

from models.random_forest_direction import X_test, X_train


# ============================================================
# PATHS
# ============================================================


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "features"
    / "company_financial_features.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "splits"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_FOLDER / "train.csv"
TEST_FILE = OUTPUT_FOLDER / "test.csv"

TARGET = "Next_Year_Revenue_Growth"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("PREPARING COMPANY ML DATA")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Companies: {df['Company'].nunique()}")
print(f"Target: {TARGET}")


# ============================================================
# SORT BY COMPANY AND YEAR
# ============================================================

df = df.sort_values(
    ["Company", "Year"]
).reset_index(drop=True)


# ============================================================
# CHECK TARGET
# ============================================================

print("\nMissing target values:")
print(df[TARGET].isna().sum())


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(
    subset=[TARGET]
).reset_index(drop=True)

# ============================================================
# HANDLE MISSING ENGINEERED FEATURES
# ============================================================

feature_cols = [
    col for col in df.columns
    if col not in ["Company", "Year", TARGET]
]

df[feature_cols] = df[feature_cols].fillna(0)

print("\nMissing values after feature cleaning:")
print(df.isna().sum().sum())

# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# Latest available year of each company = TEST
# ============================================================

test_indices = (
    df.groupby("Company")["Year"]
    .idxmax()
)

test_df = df.loc[test_indices].copy()

train_df = df.drop(test_indices).copy()


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ML DATA PREPARATION COMPLETE")
print("=" * 60)

print("\n" + "=" * 60)
print("MISSING VALUES IN TEST SET")
print("=" * 60)

print(test_df.isna().sum()[test_df.isna().sum() > 0])

print("\nMISSING VALUES IN TRAINING SET")
print("=" * 60)

print(train_df.isna().sum()[train_df.isna().sum() > 0])

print(f"\nTraining shape: {train_df.shape}")
print(f"Testing shape : {test_df.shape}")

print("\nTraining year range:")
print(f"{train_df['Year'].min()} -> {train_df['Year'].max()}")

print("\nTesting years:")
print(test_df["Year"].value_counts().sort_index())

print("\nTest companies:")
print(test_df["Company"].tolist())

print("\nSaved files:")
print(TRAIN_FILE)
print(TEST_FILE)