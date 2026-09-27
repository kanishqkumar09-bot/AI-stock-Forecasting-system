import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STOCK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "TCS_features.csv"
)

COMPANY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
    / "company_growth_ml_dataset.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AI FINANCIAL INTELLIGENCE")
print("INVESTMENT RECOMMENDATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

if not STOCK_FILE.exists():
    print("\nStock data not found!")
    exit()

if not COMPANY_FILE.exists():
    print("\nCompany data not found!")
    exit()


stock_df = pd.read_csv(STOCK_FILE)
company_df = pd.read_csv(COMPANY_FILE)


# ============================================================
# GET LATEST STOCK DATA
# ============================================================

latest = stock_df.iloc[-1]

price = latest["Close"]
volatility = latest["Volatility_20"]
rsi = latest["RSI_14"]

price_vs_sma20 = latest["Price_vs_SMA20"]
price_vs_sma50 = latest["Price_vs_SMA50"]


# ============================================================
# MARKET TREND
# ============================================================

if price_vs_sma20 > 1 and price_vs_sma50 > 1:

    trend = "BULLISH"

elif price_vs_sma20 < 1 and price_vs_sma50 < 1:

    trend = "BEARISH"

else:

    trend = "NEUTRAL"


# ============================================================
# RISK
# ============================================================

if volatility < 0.015:

    risk = "LOW"

elif volatility < 0.030:

    risk = "MODERATE"

else:

    risk = "HIGH"


# ============================================================
# COMPANY GROWTH
# ============================================================

latest_company = company_df.iloc[-1]

growth = latest_company["Next_Year_Revenue_Growth"]


# ============================================================
# RECOMMENDATION
# ============================================================

if growth >= 15 and trend == "BULLISH" and risk != "HIGH":

    recommendation = "BUY"
    confidence = "HIGH"

elif growth >= 8 and trend != "BEARISH" and risk != "HIGH":

    recommendation = "HOLD"
    confidence = "MODERATE"

elif growth >= 0 and trend == "NEUTRAL" and risk == "LOW":

    recommendation = "HOLD"
    confidence = "LOW"

else:

    recommendation = "AVOID"
    confidence = "MODERATE"


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("INVESTMENT ANALYSIS")
print("=" * 60)

print(f"\nCompany             : TCS")
print(f"Current Price       : ₹{price:.2f}")

print(f"\nExpected Growth     : {growth:.2f}%")
print(f"Market Trend        : {trend}")
print(f"Overall Risk        : {risk}")

print("\n" + "-" * 60)

print("FINAL RECOMMENDATION")
print("-" * 60)

print(f"\nRecommendation      : {recommendation}")
print(f"Confidence          : {confidence}")


# ============================================================
# REASON
# ============================================================

print("\nReason:")

if recommendation == "BUY":

    print("Strong company growth with a positive market trend.")

elif recommendation == "HOLD":

    print("The company has reasonable conditions, but signals are mixed.")

else:

    print("Growth, market trend, or risk conditions are unfavorable.")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("INVESTMENT ANALYSIS COMPLETE")
print("=" * 60)