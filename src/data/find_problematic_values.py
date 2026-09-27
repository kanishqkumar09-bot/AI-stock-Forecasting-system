import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
    / "TCS_test.csv"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

print("=" * 60)
print("FINDING PROBLEMATIC VALUES IN TCS TEST DATA")
print("=" * 60)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns


# ============================================================
# CHECK EACH COLUMN
# ============================================================

found = False

for column in numeric_columns:

    infinite_mask = np.isinf(
        df[column].to_numpy(dtype=float)
    )

    if infinite_mask.any():

        found = True

        print("\n" + "-" * 60)
        print(f"PROBLEMATIC COLUMN: {column}")
        print("-" * 60)

        problematic_rows = df.loc[
            infinite_mask,
            ["Date", column]
        ]

        print(problematic_rows)


# ============================================================
# CHECK NaN
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isna().sum()

print(
    missing[missing > 0]
)


# ============================================================
# FINAL
# ============================================================

if not found:
    print("\nNo infinity values detected using np.isinf().")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)