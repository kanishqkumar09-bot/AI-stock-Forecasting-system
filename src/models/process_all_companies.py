import pandas as pd
import numpy as np
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

SPLITS_FOLDER = STOCK_FOLDER / "splits"

SPLITS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMPANIES
# ============================================================

COMPANIES = [
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
# FEATURES
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Return_5D",
    "Return_20D",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_20",
    "Volume_Change",
    "Price_vs_SMA20",
    "Price_vs_SMA50"
]

TARGET = "Target_Next_Day_Close"


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Daily Return
    # --------------------------------------------------------

    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["Return_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Return_20D"] = (
        df["Close"].pct_change(20)
    )

    # --------------------------------------------------------
    # Moving Averages
    # --------------------------------------------------------

    df["SMA_20"] = (
        df["Close"].rolling(20).mean()
    )

    df["SMA_50"] = (
        df["Close"].rolling(50).mean()
    )

    df["EMA_20"] = (
        df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = (
        ema_12 - ema_26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Histogram"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["Volatility_20"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # Volume Change
    # --------------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    # --------------------------------------------------------
    # Price vs SMA
    # --------------------------------------------------------

    df["Price_vs_SMA20"] = (
        df["Close"] / df["SMA_20"]
    )

    df["Price_vs_SMA50"] = (
        df["Close"] / df["SMA_50"]
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df[TARGET] = (
        df["Close"].shift(-1)
    )

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    df = df.replace([np.inf,-np.inf], np.nan)
    df=df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df 


# ============================================================
# PROCESS ONE COMPANY
# ============================================================

def process_company(company):

    input_file = (
        STOCK_FOLDER
        / f"{company}.csv"
    )

    features_file = (
        STOCK_FOLDER
        / f"{company}_features.csv"
    )

    train_file = (
        SPLITS_FOLDER
        / f"{company}_train.csv"
    )

    validation_file = (
        SPLITS_FOLDER
        / f"{company}_validation.csv"
    )

    test_file = (
        SPLITS_FOLDER
        / f"{company}_test.csv"
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING: {company}")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_file.exists():

        print(f"X Input file not found:")
        print(input_file)

        return False

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = pd.read_csv(
            input_file
        )

        print(
            f"Raw rows: {len(df)}"
        )

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        required = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            print(
                f"X Missing columns: {missing}"
            )

            return False

        # ----------------------------------------------------
        # Create features
        # ----------------------------------------------------

        df = create_features(df)

        if len(df) < 100:

            print(
                f"X Not enough usable rows: {len(df)}"
            )

            return False

        # ----------------------------------------------------
        # Save feature dataset
        # ----------------------------------------------------

        df.to_csv(
            features_file,
            index=False
        )

        print(
            f"✓ Features created: {len(df)} rows"
        )

        # ----------------------------------------------------
        # Time-series split
        # ----------------------------------------------------

        n = len(df)

        train_end = int(
            n * 0.70
        )

        validation_end = int(
            n * 0.85
        )

        train = df.iloc[
            :train_end
        ].copy()

        validation = df.iloc[
            train_end:validation_end
        ].copy()

        test = df.iloc[
            validation_end:
        ].copy()

        # ----------------------------------------------------
        # Save splits
        # ----------------------------------------------------

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

        print(
            f"✓ Train: {len(train)}"
        )

        print(
            f"✓ Validation: {len(validation)}"
        )

        print(
            f"✓ Test: {len(test)}"
        )

        print(
            f"✓ Completed: {company}"
        )

        return True

    except Exception as e:

        print(
            f"X FAILED: {company}"
        )

        print(
            f"Error: {e}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 60)
print("MULTI-COMPANY STOCK PROCESSING")
print("=" * 60)

successful = []
failed = []


for company in COMPANIES:

    result = process_company(
        company
    )

    if result:

        successful.append(
            company
        )

    else:

        failed.append(
            company
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"\nSuccessfully processed: "
    f"{len(successful)}/13"
)

for company in successful:

    print(
        f"✓ {company}"
    )


print(
    f"\nMissing/failed: "
    f"{len(failed)}"
)

for company in failed:

    print(
        f"X {company}"
    )


print("\nSplits folder:")
print(SPLITS_FOLDER)

print("\n")
print("=" * 60)
print("MULTI-COMPANY PROCESSING COMPLETE")
print("=" * 60)