import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "TRAIN": PROJECT_ROOT / "data" / "processed" / "stock" / "splits" / "TCS_train.csv",
    "VALIDATION": PROJECT_ROOT / "data" / "processed" / "stock" / "splits" / "TCS_validation.csv",
    "TEST": PROJECT_ROOT / "data" / "processed" / "stock" / "splits" / "TCS_test.csv"
}


# ============================================================
# CHECK
# ============================================================

for name, file_path in FILES.items():

    print("\n" + "=" * 60)
    print(f"{name} DATA CHECK")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"\nRows: {len(df)}")

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nInfinite values:")

numeric_df = df.select_dtypes(include="number")

infinite_mask = ~numeric_df.apply(
    lambda column: column.replace(
        [float("inf"), float("-inf")],
        pd.NA
    ).notna()
).all(axis=1)

print(
    f"Rows containing problematic numeric values: "
    f"{infinite_mask.sum()}"
)

print("\nTarget missing values:")

if "Target_Next_Day_Close" in df.columns:
    print(
        df["Target_Next_Day_Close"].isnull().sum()
    )

print("\nDate range:")
print(df["Date"].min(), "→", df["Date"].max())