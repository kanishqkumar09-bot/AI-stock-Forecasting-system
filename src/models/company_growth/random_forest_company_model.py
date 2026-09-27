import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


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

MODEL_DIR = PROJECT_ROOT / "models" / "company_growth"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "random_forest_company_model.pkl"


# ============================================================
# CONFIGURATION
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

TARGET = "Next_Year_Revenue_Growth"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("COMPANY GROWTH - RANDOM FOREST")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET, "Company", "Year"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    sys.exit(1)


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace([float("inf"), float("-inf")], pd.NA)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

df = df.sort_values(["Year", "Company"]).reset_index(drop=True)

print(f"Usable rows: {len(df)}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

print("\nDATA SPLIT")
print("-" * 40)
print(f"Training rows   : {len(train_df)}")
print(f"Validation rows : {len(validation_df)}")
print(f"Test rows       : {len(test_df)}")


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# VALIDATION
# ============================================================

validation_predictions = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions
) ** 0.5

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"MAE  : {validation_mae:.4f}")
print(f"RMSE : {validation_rmse:.4f}")
print(f"R²   : {validation_r2:.4f}")


# ============================================================
# TEST
# ============================================================

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = test_df[
    ["Company", "Year", TARGET]
].copy()

results["Predicted_Growth"] = test_predictions

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results.rename(
        columns={
            TARGET: "Actual_Growth"
        }
    ).to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\n" + "=" * 60)
print("RANDOM FOREST COMPANY MODEL COMPLETE")
print("=" * 60)