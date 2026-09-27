import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    "EPS_Growth_Change"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AI FINANCIAL INTELLIGENCE")
print("COMPANY GROWTH PREDICTION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_FILE.exists():
    print("\nModel file not found!")
    print(MODEL_FILE)
    exit()


if not DATA_FILE.exists():
    print("\nDataset file not found!")
    print(DATA_FILE)
    exit()


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_FILE)

df = pd.read_csv(DATA_FILE)


# ============================================================
# SHOW COMPANIES
# ============================================================

print("\nAvailable Companies:")

companies = sorted(
    df["Company"].dropna().unique()
)

for company in companies:
    print("-", company)


# ============================================================
# USER INPUT
# ============================================================

company_name = input(
    "\nEnter company name: "
).strip()

year = int(
    input("Enter financial year: ")
)


# ============================================================
# FIND COMPANY
# ============================================================

data = df[
    (df["Company"].str.lower() == company_name.lower())
    &
    (df["Year"] == year)
]


if data.empty:

    print("\nNo data found for this company and year.")

    exit()


# ============================================================
# GET FEATURES
# ============================================================

input_data = data[FEATURES]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("COMPANY GROWTH PREDICTION")
print("=" * 60)

print(f"\nCompany              : {company_name}")
print(f"Financial Year       : {year}")
print(f"Predicted Revenue Growth : {prediction:.2f}%")


# ============================================================
# OUTLOOK
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

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)