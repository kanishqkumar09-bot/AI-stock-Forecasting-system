import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    "Infosys",
    "Wipro",
    "HCLTech",
    "Tech_Mahindra",
    "Persistent_Systems",
    "Mphasis",
    "Coforge",
    "Larsen_&_Toubro",
    "Reliance_Industries",
    "Hindustan_Unilever",
    "Bharti_Airtel",
    "KPIT_Technologies",
    "Tata_Elxsi"
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
# TRAIN ONE COMPANY
# ============================================================

def train_company(company):

    print("\n" + "=" * 70)
    print(f"TRAINING RANDOM FOREST: {company}")
    print("=" * 70)

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

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        train_file,
        validation_file,
        test_file
    ]

    for file in required_files:

        if not file.exists():

            print(
                f"❌ Missing file: {file.name}"
            )

            return False

    try:

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        train = pd.read_csv(train_file)
        validation = pd.read_csv(validation_file)
        test = pd.read_csv(test_file)

        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        for dataset_name, dataset in [
            ("Training", train),
            ("Validation", validation),
            ("Test", test)
        ]:

            missing_features = [
                feature
                for feature in FEATURES + [TARGET]
                if feature not in dataset.columns
            ]

            if missing_features:

                print(
                    f"❌ {dataset_name} missing columns:"
                )

                print(
                    missing_features
                )

                return False

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

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print(
            "Training Random Forest..."
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_predictions = (
            model.predict(X_validation)
        )

        validation_mae = (
            mean_absolute_error(
                y_validation,
                validation_predictions
            )
        )

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_validation,
                validation_predictions
            )
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_predictions = (
            model.predict(X_test)
        )

        test_mae = (
            mean_absolute_error(
                y_test,
                test_predictions
            )
        )

        test_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                test_predictions
            )
        )

        test_mape = np.mean(
            np.abs(
                (
                    y_test
                    - test_predictions
                )
                / y_test
            )
        ) * 100

        test_r2 = r2_score(
            y_test,
            test_predictions
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nVALIDATION RESULTS")
        print("-" * 50)

        print(
            f"MAE  : {validation_mae:.4f}"
        )

        print(
            f"RMSE : {validation_rmse:.4f}"
        )

        print("\nTEST RESULTS")
        print("-" * 50)

        print(
            f"MAE  : {test_mae:.4f}"
        )

        print(
            f"RMSE : {test_rmse:.4f}"
        )

        print(
            f"MAPE : {test_mape:.2f}%"
        )

        print(
            f"R²   : {test_r2:.4f}"
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance = pd.DataFrame({

            "Feature": FEATURES,

            "Importance":
                model.feature_importances_

        })

        importance = (
            importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        results = pd.DataFrame({

            "Date":
                pd.to_datetime(
                    test["Date"]
                ),

            "Actual":
                y_test.values,

            "Predicted":
                test_predictions

        })

        results_file = (
            RESULTS_FOLDER
            / f"{company}_random_forest_results.csv"
        )

        results.to_csv(
            results_file,
            index=False
        )

        # ----------------------------------------------------
        # SAVE FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance_file = (
            RESULTS_FOLDER
            / f"{company}_random_forest_feature_importance.csv"
        )

        importance.to_csv(
            importance_file,
            index=False
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_file = (
            MODEL_FOLDER
            / f"{company}_random_forest.pkl"
        )

        joblib.dump(
            model,
            model_file
        )

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        print("\n✓ Model saved:")
        print(
            model_file.name
        )

        print(
            f"✓ Results saved:"
            f" {results_file.name}"
        )

        print(
            f"✓ Feature importance saved:"
            f" {importance_file.name}"
        )

        return True

    except Exception as e:

        print(
            f"\n❌ ERROR: {company}"
        )

        print(
            f"Reason: {e}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 70)
print("     AI FINANCIAL INTELLIGENCE")
print("     MULTI-COMPANY RANDOM FOREST TRAINING")
print("=" * 70)

successful = []
failed = []


# ============================================================
# TRAIN ALL 13
# ============================================================

for company in COMPANIES:

    success = train_company(
        company
    )

    if success:

        successful.append(
            company
        )

    else:

        failed.append(
            company
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TRAINING SUMMARY")
print("=" * 70)

print(
    f"\nSuccessful: "
    f"{len(successful)}/{len(COMPANIES)}"
)

for company in successful:

    print(
        f"✓ {company}"
    )


print(
    f"\nFailed: "
    f"{len(failed)}"
)

for company in failed:

    print(
        f"❌ {company}"
    )


print("\nModels folder:")
print(MODEL_FOLDER)

print("\nResults folder:")
print(RESULTS_FOLDER)

print("\n" + "=" * 70)
print("MULTI-COMPANY RANDOM FOREST TRAINING COMPLETE")
print("=" * 70)