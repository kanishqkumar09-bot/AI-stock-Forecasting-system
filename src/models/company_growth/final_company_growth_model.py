import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
    / "company_growth_ml_dataset.csv"
)

MODEL_FOLDER = (
    PROJECT_ROOT
    / "models"
    / "company_growth"
)

MODEL_FILE = MODEL_FOLDER / "final_company_growth_model.pkl"


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

TARGET = "Next_Year_Revenue_Growth"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FINAL COMPANY GROWTH MODEL")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded.")
print("Rows:", len(df))


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna(
    subset=FEATURES + [TARGET]
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES]
y = df[TARGET]


print("Usable rows:", len(df))


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(X, y)


print("Training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\nModel saved successfully:")
print(MODEL_FILE)

print("\n" + "=" * 60)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 60)