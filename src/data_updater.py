import os
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# COMPANY TICKERS
# ============================================================

COMPANIES = {
    "Bharti_Airtel": "BHARTIARTL.NS",
    "Coforge": "COFORGE.NS",
    "HCLTech": "HCLTECH.NS",
    "Hindustan_Unilever": "HINDUNILVR.NS",
    "Infosys": "INFY.NS",
    "KPIT_Technologies": "KPITTECH.NS",
    "Larsen_&_Toubro": "LT.NS",
    "Mphasis": "MPHASIS.NS",
    "Persistent_Systems": "PERSISTENT.NS",
    "Reliance_Industries": "RELIANCE.NS",
    "Tata_Elxsi": "TATAELXSI.NS",
    "TCS": "TCS.NS",
    "Tech_Mahindra": "TECHM.NS",
    "Wipro": "WIPRO.NS",
}


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "stock"
)

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# FEATURE GENERATION
# ============================================================

def create_features(df):

    df = df.copy()

    # ========================================================
    # DAILY RETURN
    # ========================================================

    df["Daily_Return"] = (
        df["Close"].pct_change()
    )


    # ========================================================
    # MOVING AVERAGES
    # ========================================================

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

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )


    # ========================================================
    # RSI 14
    # ========================================================

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    # Avoid division-by-zero
    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI_14"] = (
        100 - (
            100 / (1 + rs)
        )
    )


    # ========================================================
    # VOLATILITY
    # ========================================================

    df["Volatility_20"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )


    # ========================================================
    # MACD
    # ========================================================

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    # ========================================================
    # MACD HISTOGRAM
    # ========================================================

    df["MACD_Histogram"] = (
        df["MACD"]
        - df["MACD_Signal"]
    )


    # ========================================================
    # 5-DAY RETURN
    # ========================================================

    df["Return_5D"] = (
        df["Close"]
        .pct_change(periods=5)
    )


    # ========================================================
    # 20-DAY RETURN
    # ========================================================

    df["Return_20D"] = (
        df["Close"]
        .pct_change(periods=20)
    )


    # ========================================================
    # VOLUME CHANGE
    # ========================================================

    if "Volume" in df.columns:

        df["Volume_Change"] = (
            df["Volume"]
            .pct_change()
        )

    else:

        df["Volume_Change"] = np.nan


    # ========================================================
    # PRICE VS SMA 20
    # ========================================================

    df["Price_vs_SMA20"] = (
        (
            df["Close"]
            - df["SMA_20"]
        )
        / df["SMA_20"]
    )


    # ========================================================
    # PRICE VS SMA 50
    # ========================================================

    df["Price_vs_SMA50"] = (
        (
            df["Close"]
            - df["SMA_50"]
        )
        / df["SMA_50"]
    )


    # ========================================================
    # PRICE CHANGE
    # ========================================================

    df["Price_Change"] = (
        df["Close"].diff()
    )


    # ========================================================
    # NEXT-DAY TARGET
    # ========================================================

    df["Target"] = (
        df["Close"].shift(-1)
    )


    # ========================================================
    # REMOVE ONLY INCOMPLETE INDICATOR ROWS
    #
    # IMPORTANT:
    # We do NOT remove the final row just because Target
    # is NaN. This allows the dashboard to use today's
    # latest data for prediction.
    # ========================================================

    feature_columns = [
        "Daily_Return",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "EMA_50",
        "RSI_14",
        "Volatility_20",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Return_5D",
        "Return_20D",
        "Volume_Change",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "Price_Change",
    ]

    df = df.dropna(
        subset=feature_columns
    )


    return df


# ============================================================
# DOWNLOAD ONE COMPANY
# ============================================================

def update_company(company_name, ticker):

    print("\n" + "=" * 60)

    print(
        f"Updating: {company_name}"
    )

    print(
        f"Ticker: {ticker}"
    )

    print("=" * 60)


    try:

        # ====================================================
        # DOWNLOAD LATEST 2 YEARS
        # ====================================================

        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )


        # ====================================================
        # CHECK DATA
        # ====================================================

        if df.empty:

            print(
                f"❌ No data received for "
                f"{company_name}"
            )

            return False


        # ====================================================
        # HANDLE MULTI-LEVEL COLUMNS
        # ====================================================

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            # Try to extract standard OHLCV names
            new_columns = []

            for column in df.columns:

                if isinstance(column, tuple):

                    if column[0] in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume"
                    ]:

                        new_columns.append(
                            column[0]
                        )

                    else:

                        new_columns.append(
                            column[-1]
                        )

                else:

                    new_columns.append(
                        column
                    )

            df.columns = new_columns


        # ====================================================
        # RESET INDEX
        # ====================================================

        df = df.reset_index()


        # ====================================================
        # MAKE SURE DATE EXISTS
        # ====================================================

        if "Date" not in df.columns:

            print(
                f"❌ Date column missing "
                f"for {company_name}"
            )

            return False


        # ====================================================
        # DATETIME CONVERSION
        # ====================================================

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )


        # ====================================================
        # REMOVE TIMEZONE
        # ====================================================

        try:

            if df["Date"].dt.tz is not None:

                df["Date"] = (
                    df["Date"]
                    .dt
                    .tz_localize(None)
                )

        except Exception:

            pass


        # ====================================================
        # REMOVE INVALID DATES
        # ====================================================

        df = df.dropna(
            subset=["Date"]
        )


        # ====================================================
        # SORT BY DATE
        # ====================================================

        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )


        # ====================================================
        # REQUIRED COLUMNS
        # ====================================================

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]


        available_columns = [
            column
            for column in required_columns
            if column in df.columns
        ]


        df = df[
            available_columns
        ]


        # ====================================================
        # CHECK CLOSE COLUMN
        # ====================================================

        if "Close" not in df.columns:

            print(
                f"❌ Close price missing "
                f"for {company_name}"
            )

            return False


        # ====================================================
        # CONVERT NUMERIC COLUMNS
        # ====================================================

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]


        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )


        # ====================================================
        # REMOVE INVALID PRICE ROWS
        # ====================================================

        df = df.dropna(
            subset=["Close"]
        )


        # ====================================================
        # CREATE FEATURES
        # ====================================================

        feature_df = create_features(
            df
        )


        # ====================================================
        # FILE PATHS
        # ====================================================

        raw_file = os.path.join(
            DATA_DIR,
            f"{company_name}.csv"
        )

        feature_file = os.path.join(
            DATA_DIR,
            f"{company_name}_features.csv"
        )


        # ====================================================
        # SAVE RAW DATA
        # ====================================================

        df.to_csv(
            raw_file,
            index=False
        )


        # ====================================================
        # SAVE FEATURE DATA
        # ====================================================

        feature_df.to_csv(
            feature_file,
            index=False
        )


        # ====================================================
        # LATEST TRADING DATA
        # ====================================================

        latest_date = (
            df["Date"].iloc[-1]
        )

        latest_close = float(
            df["Close"].iloc[-1]
        )


        # ====================================================
        # LATEST FEATURE ROW
        # ====================================================

        latest_feature_date = (
            feature_df["Date"].iloc[-1]
        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        print(
            "✅ Updated successfully"
        )

        print(
            "Latest trading date: "
            f"{latest_date.strftime('%Y-%m-%d')}"
        )

        print(
            "Latest closing price: "
            f"₹{latest_close:.2f}"
        )

        print(
            "Latest feature date: "
            f"{latest_feature_date.strftime('%Y-%m-%d')}"
        )

        print(
            f"Saved: {company_name}.csv"
        )

        print(
            f"Saved: "
            f"{company_name}_features.csv"
        )

        return True


    except Exception as e:

        print(
            f"❌ Error updating "
            f"{company_name}: {e}"
        )

        return False


# ============================================================
# UPDATE ALL COMPANIES
# ============================================================

def update_all_companies():

    print("\n")

    print("=" * 70)

    print(
        "        AI FINANCIAL INTELLIGENCE"
    )

    print(
        "        LIVE STOCK DATA UPDATE"
    )

    print("=" * 70)


    successful = 0

    failed = 0


    # ========================================================
    # UPDATE EACH COMPANY
    # ========================================================

    for company_name, ticker in COMPANIES.items():

        success = update_company(
            company_name,
            ticker
        )


        if success:

            successful += 1

        else:

            failed += 1


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "DATA UPDATE COMPLETED"
    )

    print("=" * 70)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print("=" * 70)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    update_all_companies()