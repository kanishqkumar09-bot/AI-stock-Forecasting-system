import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_ml_ready.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_return_ml_ready.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING RETURN PREDICTION DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# CREATE DIRECTION
# ============================================================

df["Target_Direction"] = (
    df["Target_Next_Day_Return"] > 0
).astype(int)


# ============================================================
# CREATE HUMAN-READABLE LABEL
# ============================================================

df["Direction_Label"] = df["Target_Direction"].map({
    0: "DOWN",
    1: "UP"
})


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\nTarget columns created:")

print("✓ Target_Next_Day_Return")
print("✓ Target_Direction")
print("✓ Direction_Label")

print("\nDirection distribution:")

print(
    df["Direction_Label"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("RETURN DATASET COMPLETE")
print("=" * 60)