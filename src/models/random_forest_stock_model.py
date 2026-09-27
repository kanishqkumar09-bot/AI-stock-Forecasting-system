import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
    / "TCS_train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
    / "TCS_validation.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
    / "TCS_test.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
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

TARGET = "Target_Next_Day_Close"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TCS RANDOM FOREST MODEL")
print("=" * 60)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_predictions
    )
)


# ============================================================
# TEST
# ============================================================

test_predictions = model.predict(
    X_test
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_mape = np.mean(
    np.abs(
        (y_test - test_predictions) / y_test
    )
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"MAE  : {validation_mae:.4f}")
print(f"RMSE : {validation_rmse:.4f}")


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"MAPE : {test_mape:.2f}%")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Date": pd.to_datetime(test["Date"]),
    "Actual": y_test.values,
    "Predicted": test_predictions
})

print("\nSample predictions:")
print("-" * 60)

print(
    results.head(10).to_string(
        index=False
    )
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

print("\nTop Feature Importances:")
print("-" * 60)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "results"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    RESULTS_FOLDER
    / "TCS_random_forest_results.csv"
)

results.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

IMPORTANCE_FILE = (
    RESULTS_FOLDER
    / "TCS_random_forest_feature_importance.csv"
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FOLDER = (
    PROJECT_ROOT
    / "models"
    / "stock_forecasting"
)

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_FOLDER
    / "TCS_random_forest.pkl"
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\nResults saved to:")
print(RESULTS_FILE)

print("\nFeature importance saved to:")
print(IMPORTANCE_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\n" + "=" * 60)
print("RANDOM FOREST COMPLETE")
print("=" * 60)