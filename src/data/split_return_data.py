import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_return_ml_ready.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "return_splits"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# TIME SPLIT
# ============================================================

train = df[
    df["Date"] < "2024-01-01"
]

validation = df[
    (df["Date"] >= "2024-01-01") &
    (df["Date"] < "2026-01-01")
]

test = df[
    df["Date"] >= "2026-01-01"
]


# ============================================================
# SAVE
# ============================================================

train.to_csv(
    OUTPUT_FOLDER / "TCS_return_train.csv",
    index=False
)

validation.to_csv(
    OUTPUT_FOLDER / "TCS_return_validation.csv",
    index=False
)

test.to_csv(
    OUTPUT_FOLDER / "TCS_return_test.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 60)
print("RETURN DATA TIME SPLIT")
print("=" * 60)

print("\nTRAIN:")
print(len(train))

print("\nVALIDATION:")
print(len(validation))

print("\nTEST:")
print(len(test))

print("\nFiles saved to:")
print(OUTPUT_FOLDER)

print("\n" + "=" * 60)
print("RETURN SPLIT COMPLETE")
print("=" * 60)