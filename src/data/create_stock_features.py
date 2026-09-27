import pandas as pd
import numpy as np
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
    / "TCS.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
)

OUTPUT_FILE = OUTPUT_FOLDER / "TCS_features.csv"


# ============================================================
# LOAD CLEAN DATA
# ============================================================

print("=" * 60)
print("CREATING TCS STOCK FEATURES")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# DAILY RETURNS
# ============================================================

df["Daily_Return"] = df["Close"].pct_change()


# ============================================================
# MULTI-DAY RETURNS
# ============================================================

df["Return_5D"] = df["Close"].pct_change(periods=5)

df["Return_20D"] = df["Close"].pct_change(periods=20)


# ============================================================
# SIMPLE MOVING AVERAGES
# ============================================================

df["SMA_20"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

df["SMA_50"] = (
    df["Close"]
    .rolling(window=50)
    .mean()
)


# ============================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================

df["EMA_20"] = (
    df["Close"]
    .ewm(span=20, adjust=False)
    .mean()
)


# ============================================================
# RSI - 14 PERIOD
# ============================================================

delta = df["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

average_gain = (
    gain
    .rolling(window=14)
    .mean()
)

average_loss = (
    loss
    .rolling(window=14)
    .mean()
)

rs = average_gain / average_loss.replace(0, np.nan)

df["RSI_14"] = 100 - (
    100 / (1 + rs)
)


# ============================================================
# MACD
# ============================================================

ema_12 = (
    df["Close"]
    .ewm(span=12, adjust=False)
    .mean()
)

ema_26 = (
    df["Close"]
    .ewm(span=26, adjust=False)
    .mean()
)

df["MACD"] = ema_12 - ema_26

df["MACD_Signal"] = (
    df["MACD"]
    .ewm(span=9, adjust=False)
    .mean()
)

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)


# ============================================================
# VOLATILITY
# ============================================================

df["Volatility_20"] = (
    df["Daily_Return"]
    .rolling(window=20)
    .std()
)


# ============================================================
# VOLUME CHANGE
# ============================================================
previous_volume = df["Volume"].shift(1)
df["Volume_Change"] = np.where(previous_volume>0, (df["Volume"] - previous_volume) / previous_volume, np.nan)

# ============================================================
# PRICE VS MOVING AVERAGE
# ============================================================

df["Price_vs_SMA20"] = (
    df["Close"] / df["SMA_20"]
)

df["Price_vs_SMA50"] = (
    df["Close"] / df["SMA_50"]
)


# ============================================================
# REMOVE INITIAL NaN ROWS
# ============================================================
df=df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\nFeature engineering completed.")

print("\nDataset shape:")
print(df.shape)

print("\nFeatures created:")

feature_columns = [
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

for feature in feature_columns:
    print(f"  ✓ {feature}")


print("\nSaved to:")
print(OUTPUT_FILE)

print("\nLast 5 rows:")
print(df.tail())

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)