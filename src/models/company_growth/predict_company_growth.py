import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "company_growth"
    / "final_company_growth_model.pkl"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
    / "company_growth_ml_dataset.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Revenue_Growth",
    "Profit_Growth",
    "EPS_Growth",
    "EBIT_Margin",
    "Net_Profit_Margin",
    "Revenue_Growth_Change",
    "Profit_Growth_Change",
    "EPS_Growth_Change",
]


# ============================================================
# COMPANY
# ============================================================

COMPANY = "TCS"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI FINANCIAL INTELLIGENCE")
print("COMPANY GROWTH PREDICTION")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_FILE.exists():

    print("\nERROR: Model file not found:")
    print(MODEL_FILE)

    raise SystemExit


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    print("\nERROR: Dataset file not found:")
    print(DATA_FILE)

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("\nModel loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded: {len(df)} rows")


# ============================================================
# CHECK COMPANY COLUMN
# ============================================================

if "Company" not in df.columns:

    print("\nERROR: 'Company' column not found.")

    raise SystemExit


# ============================================================
# FIND COMPANY
# ============================================================

company_data = df[
    df["Company"]
    .astype(str)
    .str.strip()
    .str.lower()
    == COMPANY.lower()
].copy()


if company_data.empty:

    print(f"\nERROR: No data found for {COMPANY}")

    print("\nAvailable companies:")

    companies = sorted(
        df["Company"]
        .dropna()
        .astype(str)
        .unique()
    )

    for name in companies:
        print(" -", name)

    raise SystemExit


# ============================================================
# FIND LATEST FINANCIAL YEAR
# ============================================================

company_data = company_data.sort_values("Year")

latest_year = company_data["Year"].iloc[-1]


latest_data = company_data[
    company_data["Year"] == latest_year
].copy()


# ============================================================
# DISPLAY COMPANY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("COMPANY INFORMATION")
print("=" * 70)

print(f"\nCompany               : {COMPANY}")
print(f"Latest Financial Year : {latest_year}")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in latest_data.columns
]


if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise SystemExit


# ============================================================
# PREPARE INPUT DATA
# ============================================================

input_data = latest_data[FEATURES].copy()


# Replace infinite values

input_data = input_data.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# Remove missing values

input_data = input_data.dropna()


if input_data.empty:

    print("\nERROR: Financial features contain missing values.")

    raise SystemExit


# ============================================================
# DISPLAY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FINANCIAL FEATURES USED")
print("=" * 70)


for feature in FEATURES:

    value = input_data.iloc[0][feature]

    print(f"{feature:<30}: {value:.4f}")


# ============================================================
# MAKE PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("COMPANY GROWTH PREDICTION")
print("=" * 70)

print(f"\nCompany              : {COMPANY}")
print(f"Financial Year       : {latest_year}")
print(f"Predicted Growth     : {prediction:.2f}%")


# ============================================================
# GROWTH OUTLOOK
# ============================================================

if prediction >= 15:

    outlook = "HIGH GROWTH"

elif prediction >= 8:

    outlook = "MODERATE GROWTH"

elif prediction >= 0:

    outlook = "LOW GROWTH"

else:

    outlook = "NEGATIVE GROWTH"


print(f"Growth Outlook       : {outlook}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)