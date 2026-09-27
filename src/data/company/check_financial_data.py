import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "raw"
    / "company_financials.csv"
)


print("=" * 60)
print("COMPANY FINANCIAL DATA CHECK")
print("=" * 60)


df = pd.read_csv(FILE)


print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nMissing values:")
print(df.isna().sum())


print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\nYear range:")
print(
    df["Year"].min(),
    "→",
    df["Year"].max()
)


print("\nCompany count:")
print(
    df["Company"].nunique()
)


print("\nFinancial data:")
print(df.to_string(index=False))


print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)