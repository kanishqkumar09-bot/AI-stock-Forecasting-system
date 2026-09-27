import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "ml"
    / "company_growth_ml_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "company_growth"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "company"
    / "results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("COMPANY GROWTH - LINEAR REGRESSION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows loaded: {len(df)}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

features = [
    "Revenue_Growth",
    "Profit_Growth",
    "EPS_Growth",
    "EBIT_Margin",
    "Net_Profit_Margin",
    "Revenue_Growth_Change",
    "Profit_Growth_Change",
    "EPS_Growth_Change"
]

target = "Next_Year_Revenue_Growth"


X = df[features].copy()
y = df[target].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

valid_rows = (
    X.notna().all(axis=1)
    & y.notna()
)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print(f"Usable rows: {len(X)}")


# ============================================================
# TIME-BASED SPLIT
# ============================================================
# IMPORTANT:
# Financial data should be split chronologically.
# We do NOT randomly shuffle the data.

data = df.loc[valid_rows].copy()

data = data.sort_values(
    ["Year", "Company"]
).reset_index(drop=True)

X = data[features]
y = data[target]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\nDATA SPLIT")
print("-" * 60)

print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_validation)}")
print(f"Test rows       : {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression...")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

validation_predictions = model.predict(
    X_validation
)

test_predictions = model.predict(
    X_test
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(actual, predicted, name):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + "=" * 60)
    print(f"{name.upper()} RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# VALIDATION
# ============================================================

validation_metrics = evaluate_model(
    y_validation,
    validation_predictions,
    "Validation"
)


# ============================================================
# TEST
# ============================================================

test_metrics = evaluate_model(
    y_test,
    test_predictions,
    "Test"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = data.loc[
    X_test.index,
    ["Company", "Year"]
].copy()

results["Actual_Growth"] = y_test.values

results["Predicted_Growth"] = test_predictions

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results.to_string(index=False)
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

coefficients["Absolute_Impact"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Impact",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE COEFFICIENTS")
print("=" * 60)

print(
    coefficients[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULT_DIR
    / "linear_regression_company_results.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR
    / "linear_regression_company_growth.pkl"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# FINAL
# ============================================================

print("\nResults saved to:")
print(results_file)

print("\nModel saved to:")
print(model_file)

print("\n" + "=" * 60)
print("LINEAR REGRESSION COMPANY MODEL COMPLETE")
print("=" * 60)