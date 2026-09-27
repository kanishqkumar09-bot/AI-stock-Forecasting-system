from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# ============================================================
# AI FINANCIAL INTELLIGENCE
# TCS NEXT-DAY STOCK PRICE PREDICTION
# XGBOOST MODEL
# ============================================================

print("=" * 70)
print("AI FINANCIAL INTELLIGENCE")
print("NEXT-DAY STOCK PRICE PREDICTION")
print("=" * 70)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# COMPANY
# ============================================================

COMPANY = "TCS"


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stock_forecasting"
    / "TCS_xgboost.json"
)


# ============================================================
# FEATURE DATA PATH
# ============================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_features.csv"
)


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


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

print("\nLoading XGBoost model...")

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = XGBRegressor()

    model.load_model(MODEL_PATH)

    print("XGBoost model loaded successfully.")

except Exception as e:

    print("\nERROR: Could not load XGBoost model.")
    print(f"Reason: {e}")
    raise SystemExit(1)


# ============================================================
# LOAD LATEST STOCK DATA
# ============================================================

print("\nLoading latest stock data...")

try:

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    stock_data = pd.read_csv(FEATURE_FILE)

    if stock_data.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    # Convert Date
    stock_data["Date"] = pd.to_datetime(
        stock_data["Date"]
    )

    # Sort by date
    stock_data = (
        stock_data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Check required features
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in stock_data.columns
    ]

    if missing_features:

        print("\nERROR: Missing features:")

        for feature in missing_features:
            print(f" - {feature}")

        raise SystemExit(1)

    # Latest row
    latest = stock_data.iloc[-1]

    latest_date = latest["Date"]

    current_price = float(
        latest["Close"]
    )

    # Create model input
    input_data = pd.DataFrame(
        [[latest[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    # Check missing values
    if input_data.isnull().any().any():

        print("\nERROR: Missing values found in latest data.")

        print(
            input_data.isnull()
            .sum()
        )

        raise SystemExit(1)

    print("Latest stock data loaded successfully.")

except Exception as e:

    print("\nERROR: Could not load stock data.")
    print(f"Reason: {e}")
    raise SystemExit(1)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating next-day prediction...")

try:

    prediction = model.predict(
        input_data
    )[0]

    prediction = float(prediction)

except Exception as e:

    print("\nERROR: Prediction failed.")
    print(f"Reason: {e}")
    raise SystemExit(1)


# ============================================================
# EXPECTED CHANGE
# ============================================================

expected_change = (
    (prediction - current_price)
    / current_price
) * 100


absolute_change = (
    prediction - current_price
)


# ============================================================
# STOCK OUTLOOK
# ============================================================

if expected_change > 0.5:

    outlook = "Potential UPWARD movement"

elif expected_change < -0.5:

    outlook = "Potential DOWNWARD movement"

else:

    outlook = "Potential SIDEWAYS movement"


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STOCK PRICE PREDICTION")
print("=" * 70)

print(
    f"Company              : {COMPANY}"
)

print(
    f"Latest Available Date: "
    f"{latest_date.strftime('%Y-%m-%d')}"
)

print(
    f"Current Price        : "
    f"₹{current_price:,.2f}"
)

print(
    f"Predicted Next-Day   : "
    f"₹{prediction:,.2f}"
)

print(
    f"Expected Change      : "
    f"{expected_change:+.2f}%"
)

print(
    f"Expected Movement    : "
    f"₹{absolute_change:+,.2f}"
)


print("\nStock Outlook:")
print(outlook)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)