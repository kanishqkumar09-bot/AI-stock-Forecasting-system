import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
    / "return_splits"
    / "TCS_return_train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "return_splits"
    / "TCS_return_validation.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "return_splits"
    / "TCS_return_test.csv"
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

TARGET = "Target_Direction"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TCS XGBOOST - DIRECTION PREDICTION")
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
# CLASS BALANCE
# ============================================================

positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\nTraining class distribution:")
print(f"DOWN: {negative_count}")
print(f"UP  : {positive_count}")

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost Classifier...")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_validation, y_validation)
    ],
    verbose=False
)


# ============================================================
# VALIDATION
# ============================================================

validation_predictions = model.predict(
    X_validation
)

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


# ============================================================
# TEST
# ============================================================

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy : {validation_accuracy:.4f}")
print(f"F1 Score : {validation_f1:.4f}")


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")


print("\n" + "=" * 60)
print("MAJORITY BASELINE")
print("=" * 60)

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)


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
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Date": pd.to_datetime(test["Date"]),
    "Actual_Direction": y_test.values,
    "Predicted_Direction": test_predictions,
    "UP_Probability": test_probabilities
})

results["Actual_Label"] = results[
    "Actual_Direction"
].map({
    0: "DOWN",
    1: "UP"
})

results["Predicted_Label"] = results[
    "Predicted_Direction"
].map({
    0: "DOWN",
    1: "UP"
})


print("\nSample predictions:")
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
    / "TCS_xgboost_direction_results.csv"
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
    / "TCS_xgboost_direction_importance.csv"
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
    / "TCS_xgboost_direction.json"
)

model.save_model(
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
print("XGBOOST DIRECTION MODEL COMPLETE")
print("=" * 60)