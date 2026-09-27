import yfinance as yf
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMPANIES
# ============================================================

companies = {

    "TCS": "TCS.NS",

    "Infosys": "INFY.NS",

    "Wipro": "WIPRO.NS",

    "HCLTech": "HCLTECH.NS",

    "Tech_Mahindra": "TECHM.NS",

    "Bharti_Airtel": "BHARTIARTL.NS",

    "Reliance_Industries": "RELIANCE.NS",

    "Larsen_&_Toubro": "LT.NS",

    "LTIMindtree": "LTIM.NS",

    "Mphasis": "MPHASIS.NS",

    "Persistent_Systems": "PERSISTENT.NS",

    "Coforge": "COFORGE.NS",

    "KPIT_Technologies": "KPITTECH.NS",

    "Tata_Elxsi": "TATAELXSI.NS",

    "Hindustan_Unilever": "HINDUNILVR.NS"

}


# ============================================================
# DOWNLOAD DATA
# ============================================================

print("=" * 70)
print("AI FINANCIAL INTELLIGENCE")
print("STOCK DATA DOWNLOAD")
print("=" * 70)


for company, ticker in companies.items():

    print("\n" + "-" * 70)

    print(f"Downloading: {company}")

    print(f"Ticker: {ticker}")

    try:

        # Download approximately 5 years of data
        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=False,
            progress=False
        )

        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------

        if data.empty:

            print(
                f"❌ No data found for {company}"
            )

            continue

        # ----------------------------------------------------
        # FIX COLUMN FORMAT
        # ----------------------------------------------------

        if hasattr(data.columns, "levels"):

            data.columns = data.columns.get_level_values(0)

        # ----------------------------------------------------
        # RESET INDEX
        # ----------------------------------------------------

        data = data.reset_index()

        # ----------------------------------------------------
        # KEEP REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        data = data[
            [
                column
                for column in required_columns
                if column in data.columns
            ]
        ]

        # ----------------------------------------------------
        # REMOVE MISSING VALUES
        # ----------------------------------------------------

        data = data.dropna()

        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        output_file = (
            OUTPUT_FOLDER
            / f"{company}.csv"
        )

        data.to_csv(
            output_file,
            index=False
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        print(
            f"✅ Downloaded {len(data)} rows"
        )

        print(
            f"Saved to: {output_file}"
        )

    except Exception as e:

        print(
            f"❌ Error downloading {company}"
        )

        print(e)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STOCK DATA DOWNLOAD COMPLETE")
print("=" * 70)

print("\nFiles saved in:")

print(OUTPUT_FOLDER)