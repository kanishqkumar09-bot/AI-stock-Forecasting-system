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


# ============================================================
# SETTINGS
# ============================================================

FEATURE_PATTERN = "*_features.csv"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CREATING FORECASTING TARGETS FOR ALL COMPANIES")
print("=" * 70)


# ============================================================
# FIND FEATURE FILES
# ============================================================

feature_files = list(
    STOCK_FOLDER.glob(FEATURE_PATTERN)
)


if not feature_files:

    raise FileNotFoundError(
        "No *_features.csv files found."
    )


print(
    f"\nFound {len(feature_files)} feature files."
)


# ============================================================
# PROCESS COMPANIES
# ============================================================

successful = 0
failed = 0


for feature_file in feature_files:

    company = feature_file.stem.replace(
        "_features",
        ""
    )

    print("\n" + "-" * 70)
    print(f"PROCESSING: {company}")
    print("-" * 70)


    try:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        df = pd.read_csv(
            feature_file
        )


        if df.empty:

            print(
                "❌ Dataset is empty."
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "Date",
            "Close"
        ]


        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]


        if missing_columns:

            print(
                f"❌ Missing columns: "
                f"{missing_columns}"
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        df["Date"] = pd.to_datetime(
            df["Date"]
        )


        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # NEXT-DAY CLOSE TARGET
        # ----------------------------------------------------

        df["Target_Next_Day_Close"] = (
            df["Close"].shift(-1)
        )


        # ----------------------------------------------------
        # NEXT-DAY RETURN TARGET
        # ----------------------------------------------------

        df["Target_Next_Day_Return"] = (
            df["Close"].shift(-1)
            / df["Close"]
            - 1
        )


        # ----------------------------------------------------
        # REMOVE LAST ROW
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "Target_Next_Day_Close",
                "Target_Next_Day_Return"
            ]
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # SAVE BACK TO FEATURE FILE
        # ----------------------------------------------------

        df.to_csv(
            feature_file,
            index=False
        )


        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        print(
            "✓ Target_Next_Day_Close created"
        )

        print(
            "✓ Target_Next_Day_Return created"
        )

        print(
            f"Rows: {len(df)}"
        )

        print(
            f"Date: "
            f"{df['Date'].min().date()} "
            f"→ "
            f"{df['Date'].max().date()}"
        )

        print(
            f"Saved: {feature_file.name}"
        )

        print("✅ SUCCESS")

        successful += 1


    except Exception as e:

        print(
            f"❌ ERROR: {e}"
        )

        failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("TARGET CREATION COMPLETE")
print("=" * 70)

print(
    f"Successful : {successful}"
)

print(
    f"Failed     : {failed}"
)

print(
    f"\nFeature files updated in:"
)

print(
    STOCK_FOLDER
)

print("=" * 70)