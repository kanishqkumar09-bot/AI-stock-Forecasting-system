import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
    / "TCS_test.csv"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("TCS BASELINE MODEL")
print("=" * 60)

df = pd.read_csv(TEST_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# CREATE BASELINE PREDICTION
# ============================================================

# Today's closing price is used as tomorrow's prediction.

df["Baseline_Prediction"] = df["Close"]


# ============================================================
# ACTUAL TARGET
# ============================================================

actual = df["Target_Next_Day_Close"]

predicted = df["Baseline_Prediction"]


# ============================================================
# EVALUATION METRICS
# ============================================================

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


# ============================================================
# MAPE
# ============================================================

mape = np.mean(
    np.abs(
        (actual - predicted) / actual
    )
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\nBaseline Results")
print("-" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")
print("-" * 60)

results = pd.DataFrame({
    "Date": df["Date"],
    "Actual": actual,
    "Predicted": predicted
})

print(
    results.head(10).to_string(
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
    / "TCS_baseline_results.csv"
)

results.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nResults saved to:")
print(RESULTS_FILE)

print("\n" + "=" * 60)
print("BASELINE MODEL COMPLETE")
print("=" * 60)