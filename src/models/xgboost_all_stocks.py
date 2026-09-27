import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
import joblib


# ============================================================
# AI FINANCIAL INTELLIGENCE
# XGBOOST - ALL STOCKS
# ============================================================

print("=" * 70)
print("AI FINANCIAL INTELLIGENCE")
print("XGBOOST STOCK MODEL TRAINING")
print("=" * 70)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DIRECTORIES
# ============================================================

SPLITS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "splits"
)

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stock"
    / "results"
)

MODEL_FOLDER = (
    PROJECT_ROOT
    / "models"
    / "stock_forecasting"
)

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COMPANIES
# ============================================================

COMPANIES = [
    "Bharti_Airtel",
    "Coforge",
    "HCLTech",
    "Hindustan_Unilever",
    "Infosys",
    "KPIT_Technologies",
    "Larsen_&_Toubro",
    "Mphasis",
    "Persistent_Systems",
    "Reliance_Industries",
    "Tata_Elxsi",
    "TCS",
    "Tech_Mahindra",
    "Wipro"
]


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
# RESULTS STORAGE
# ============================================================

all_results = []


# ============================================================
# TRAIN EACH COMPANY
# ============================================================

for company in COMPANIES:

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {company}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # FILE PATHS
        # ----------------------------------------------------

        train_file = (
            SPLITS_FOLDER
            / f"{company}_train.csv"
        )

        validation_file = (
            SPLITS_FOLDER
            / f"{company}_validation.csv"
        )

        test_file = (
            SPLITS_FOLDER
            / f"{company}_test.csv"
        )


        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not train_file.exists():
            print(f"❌ Training file missing: {train_file}")
            continue

        if not validation_file.exists():
            print(f"❌ Validation file missing: {validation_file}")
            continue

        if not test_file.exists():
            print(f"❌ Test file missing: {test_file}")
            continue


        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        train = pd.read_csv(train_file)
        validation = pd.read_csv(validation_file)
        test = pd.read_csv(test_file)


        # ----------------------------------------------------
        # PREPARE DATA
        # ----------------------------------------------------

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_validation = validation[FEATURES]
        y_validation = validation[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]


        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print("Training XGBoost...")

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_validation, y_validation)
            ],
            verbose=False
        )


        # ----------------------------------------------------
        # VALIDATION PREDICTION
        # ----------------------------------------------------

        validation_predictions = model.predict(
            X_validation
        )

        validation_mae = np.mean(
            np.abs(
                y_validation.values
                - validation_predictions
            )
        )

        validation_rmse = np.sqrt(
            np.mean(
                (
                    y_validation.values
                    - validation_predictions
                ) ** 2
            )
        )


        # ----------------------------------------------------
        # TEST PREDICTION
        # ----------------------------------------------------

        test_predictions = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # TEST METRICS
        # ----------------------------------------------------

        test_mae = np.mean(
            np.abs(
                y_test.values
                - test_predictions
            )
        )

        test_rmse = np.sqrt(
            np.mean(
                (
                    y_test.values
                    - test_predictions
                ) ** 2
            )
        )

        test_mape = np.mean(
            np.abs(
                (
                    y_test.values
                    - test_predictions
                )
                / y_test.values
            )
        ) * 100


        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance = pd.DataFrame({

            "Feature": FEATURES,

            "Importance":
                model.feature_importances_

        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )


        # ----------------------------------------------------
        # SAVE FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance_file = (
            RESULTS_FOLDER
            / f"{company}_xgboost_feature_importance.csv"
        )

        importance.to_csv(
            importance_file,
            index=False
        )


        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_results = pd.DataFrame({

            "Date":
                pd.to_datetime(
                    test["Date"]
                ),

            "Actual":
                y_test.values,

            "Predicted":
                test_predictions

        })


        prediction_file = (
            RESULTS_FOLDER
            / f"{company}_xgboost_results.csv"
        )

        prediction_results.to_csv(
            prediction_file,
            index=False
        )


        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_file = (
            MODEL_FOLDER
            / f"{company}_xgboost.json"
        )

        model.save_model(
            model_file
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_results.append({

            "Company": company,

            "Validation_MAE":
                validation_mae,

            "Validation_RMSE":
                validation_rmse,

            "Test_MAE":
                test_mae,

            "Test_RMSE":
                test_rmse,

            "Test_MAPE":
                test_mape

        })


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        print("\nResults:")

        print(
            f"Validation MAE  : "
            f"{validation_mae:.4f}"
        )

        print(
            f"Validation RMSE : "
            f"{validation_rmse:.4f}"
        )

        print(
            f"Test MAE        : "
            f"{test_mae:.4f}"
        )

        print(
            f"Test RMSE       : "
            f"{test_rmse:.4f}"
        )

        print(
            f"Test MAPE       : "
            f"{test_mape:.2f}%"
        )

        print(
            f"Model saved     : "
            f"{model_file.name}"
        )


        # ----------------------------------------------------
        # TOP FEATURES
        # ----------------------------------------------------

        print("\nTop 5 Features:")

        print(
            importance
            .head(5)
            .to_string(index=False)
        )


    except Exception as e:

        print(
            f"\n❌ ERROR training {company}"
        )

        print(
            f"Reason: {e}"
        )


# ============================================================
# FINAL COMPARISON
# ============================================================

if all_results:

    comparison = pd.DataFrame(
        all_results
    )


    comparison = comparison.sort_values(
        "Test_MAPE"
    )


    comparison_file = (
        RESULTS_FOLDER
        / "xgboost_all_companies_comparison.csv"
    )


    comparison.to_csv(
        comparison_file,
        index=False
    )


    print("\n")
    print("=" * 70)
    print("XGBOOST MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )


    print("\n")
    print("=" * 70)
    print("BEST PERFORMING COMPANIES")
    print("=" * 70)

    print(
        comparison[
            [
                "Company",
                "Test_MAPE"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


    print("\nComparison saved to:")
    print(comparison_file)


else:

    print("\n❌ No models were successfully trained.")


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("ALL XGBOOST TRAINING COMPLETE")
print("=" * 70)