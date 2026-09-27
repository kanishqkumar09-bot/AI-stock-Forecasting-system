import pandas as pd
import numpy as np
from pathlib import Path
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "stock_forecasting"
    / "TCS_random_forest.pkl"
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
# HEADER
# ============================================================

print("=" * 60)
print("        TCS AI STOCK PREDICTION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print("\n✓ Random Forest model loaded")


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("✓ TCS feature data loaded")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# GET LATEST VALID ROW
# ============================================================

latest_data = df.dropna(
    subset=FEATURES
).iloc[-1]


# ============================================================
# PREPARE INPUT
# ============================================================

X_latest = pd.DataFrame(
    [latest_data[FEATURES].values],
    columns=FEATURES
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    X_latest
)[0]

latest_close = latest_data["Close"]

expected_change = (
    (prediction - latest_close)
    / latest_close
) * 100


# ============================================================
# SIGNAL
# ============================================================

if expected_change > 1:
    signal = "BULLISH"

elif expected_change < -1:
    signal = "BEARISH"

else:
    signal = "NEUTRAL"


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"\nLatest Date       : "
    f"{latest_data['Date'].date()}"
)

print(
    f"Latest Close      : "
    f"₹{latest_close:.2f}"
)

print(
    f"Predicted Next Day: "
    f"₹{prediction:.2f}"
)

print(
    f"Expected Change   : "
    f"{expected_change:+.2f}%"
)

print(
    f"Signal            : "
    f"{signal}"
)

print(
    "\nModel             : "
    "Random Forest"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)