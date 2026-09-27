import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
)

OUTPUT_FOLDER = (
    STOCK_FOLDER
    / "splits"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMPANIES
# ============================================================

COMPANIES = [
    "TCS",
    "Infosys",
    "Wipro",
    "HCLTech",
    "Tech_Mahindra",
    "Persistent_Systems",
    "Mphasis",
    "Coforge",
    "Larsen_&_Toubro",
    "Reliance_Industries",
    "Hindustan_Unilever",
    "Bharti_Airtel",
    "KPIT_Technologies",
    "Tata_Elxsi"
]


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_company(company):

    input_file = (
        STOCK_FOLDER
        / f"{company}_ml_ready.csv"
    )

    print("\n" + "=" * 60)
    print(f"{company} TIME-SERIES DATA SPLIT")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not input_file.exists():

        print(
            f"WARNING: {company}_ml_ready.csv not found."
        )

        print(
            f"Expected: {input_file}"
        )

        return False

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    if "Date" not in df.columns:
        print(
            f"ERROR: Date column missing for {company}"
        )
        return False

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # DATA RANGE
    # --------------------------------------------------------

    print("\nFULL DATASET")
    print("-" * 60)

    print(
        f"Start: {df['Date'].min().date()}"
    )

    print(
        f"End:   {df['Date'].max().date()}"
    )

    print(
        f"Rows:  {len(df)}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    # 70% Training
    # 15% Validation
    # 15% Testing

    n = len(df)

    train_end = int(n * 0.70)

    validation_end = int(n * 0.85)

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(train) == 0:
        print("ERROR: Training dataset is empty.")
        return False

    if len(validation) == 0:
        print("ERROR: Validation dataset is empty.")
        return False

    if len(test) == 0:
        print("ERROR: Test dataset is empty.")
        return False

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    train_file = (
        OUTPUT_FOLDER
        / f"{company}_train.csv"
    )

    validation_file = (
        OUTPUT_FOLDER
        / f"{company}_validation.csv"
    )

    test_file = (
        OUTPUT_FOLDER
        / f"{company}_test.csv"
    )

    train.to_csv(
        train_file,
        index=False
    )

    validation.to_csv(
        validation_file,
        index=False
    )

    test.to_csv(
        test_file,
        index=False
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\nSPLIT RESULTS")

    print("-" * 60)

    print("\nTRAINING")

    print(
        f"Rows: {len(train)}"
    )

    print(
        f"Date: {train['Date'].min().date()} "
        f"→ {train['Date'].max().date()}"
    )

    print("\nVALIDATION")

    print(
        f"Rows: {len(validation)}"
    )

    print(
        f"Date: {validation['Date'].min().date()} "
        f"→ {validation['Date'].max().date()}"
    )

    print("\nTEST")

    print(
        f"Rows: {len(test)}"
    )

    print(
        f"Date: {test['Date'].min().date()} "
        f"→ {test['Date'].max().date()}"
    )

    print("\nFiles saved.")

    return True


# ============================================================
# RUN ALL COMPANIES
# ============================================================

print("=" * 60)
print("MULTI-COMPANY TIME-SERIES SPLIT")
print("=" * 60)

successful = []
failed = []


for company in COMPANIES:

    success = split_company(company)

    if success:
        successful.append(company)
    else:
        failed.append(company)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"\nSuccessfully processed: "
    f"{len(successful)}/{len(COMPANIES)}"
)

for company in successful:
    print(f"  ✓ {company}")


if failed:

    print(
        f"\nMissing/failed: "
        f"{len(failed)}"
    )

    for company in failed:
        print(f"  ✗ {company}")


print("\nSplits folder:")
print(OUTPUT_FOLDER)

print("\n" + "=" * 60)
print("MULTI-COMPANY SPLIT COMPLETE")
print("=" * 60)