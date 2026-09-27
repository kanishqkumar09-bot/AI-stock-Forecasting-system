import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_features.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AI FINANCIAL INTELLIGENCE")
print("STOCK RISK ANALYSIS")
print("=" * 60)


# ============================================================
# CHECK FILE
# ============================================================

if not DATA_FILE.exists():

    print("\nStock feature file not found!")
    print(DATA_FILE)
    exit()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nStock data loaded successfully.")


# ============================================================
# GET LATEST DATA
# ============================================================

latest = df.iloc[-1]


volatility = latest["Volatility_20"]
rsi = latest["RSI_14"]
price_sma20 = latest["Price_vs_SMA20"]
price_sma50 = latest["Price_vs_SMA50"]


# ============================================================
# VOLATILITY RISK
# ============================================================

if volatility < 0.015:

    volatility_risk = "LOW"

elif volatility < 0.030:

    volatility_risk = "MODERATE"

else:

    volatility_risk = "HIGH"


# ============================================================
# RSI RISK
# ============================================================

if rsi > 70:

    rsi_status = "OVERBOUGHT"

elif rsi < 30:

    rsi_status = "OVERSOLD"

else:

    rsi_status = "NORMAL"


# ============================================================
# TREND
# ============================================================

if price_sma20 > 1 and price_sma50 > 1:

    trend = "BULLISH"

elif price_sma20 < 1 and price_sma50 < 1:

    trend = "BEARISH"

else:

    trend = "NEUTRAL"


# ============================================================
# OVERALL RISK
# ============================================================

risk_score = 0


if volatility_risk == "HIGH":

    risk_score += 2

elif volatility_risk == "MODERATE":

    risk_score += 1


if rsi_status == "OVERBOUGHT":

    risk_score += 1

elif rsi_status == "OVERSOLD":

    risk_score += 1


if trend == "BEARISH":

    risk_score += 1


if risk_score >= 4:

    overall_risk = "HIGH"

elif risk_score >= 2:

    overall_risk = "MODERATE"

else:

    overall_risk = "LOW"


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("RISK ANALYSIS")
print("=" * 60)

print(f"\nLatest Date       : {latest['Date']}")
print(f"Closing Price     : ₹{latest['Close']:.2f}")

print(f"\n20-Day Volatility : {volatility:.4f}")
print(f"Volatility Risk   : {volatility_risk}")

print(f"\nRSI (14)          : {rsi:.2f}")
print(f"RSI Status        : {rsi_status}")

print(f"\nPrice vs SMA20    : {price_sma20:.4f}")
print(f"Price vs SMA50    : {price_sma50:.4f}")

print(f"\nMarket Trend      : {trend}")

print("\n" + "-" * 60)

print(f"Overall Risk      : {overall_risk}")

print("-" * 60)


# ============================================================
# COMPLETE
# ============================================================

print("\nRISK ANALYSIS COMPLETE")
print("=" * 60)