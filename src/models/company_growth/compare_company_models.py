import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "company" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULTS_DIR / "company_model_comparison.csv"


# ============================================================
# FEATURES / TARGET
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

print("=" * 70)
print("COMPANY GROWTH - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print(f"\nRows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

df = df.sort_values(
    ["Year", "Company"]
).reset_index(drop=True)

print(f"Usable rows: {len(df)}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end]
validation_df = df.iloc[train_end:validation_end]
test_df = df.iloc[validation_end:]


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


print("\nDATA SPLIT")
print("-" * 50)

print(f"Training rows   : {len(train_df)}")
print(f"Validation rows : {len(validation_df)}")
print(f"Test rows       : {len(test_df)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# MODEL EVALUATION
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_prediction = model.predict(
        X_validation
    )

    validation_mae = mean_absolute_error(
        y_validation,
        validation_prediction
    )

    validation_rmse = mean_squared_error(
        y_validation,
        validation_prediction
    ) ** 0.5

    validation_r2 = r2_score(
        y_validation,
        validation_prediction
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_prediction = model.predict(
        X_test
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    test_rmse = mean_squared_error(
        y_test,
        test_prediction
    ) ** 0.5

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Validation_MAE": validation_mae,
        "Validation_RMSE": validation_rmse,
        "Validation_R2": validation_r2,

        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2
    })

    print("\nValidation:")
    print(f"MAE  : {validation_mae:.4f}")
    print(f"RMSE : {validation_rmse:.4f}")
    print(f"R²   : {validation_r2:.4f}")

    print("\nTest:")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"R²   : {test_r2:.4f}")


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Test_MAE"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison.iloc[0]["Model"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Selected model: {best_model}")

print(
    f"Test MAE : "
    f"{comparison.iloc[0]['Test_MAE']:.4f}"
)

print(
    f"Test RMSE: "
    f"{comparison.iloc[0]['Test_RMSE']:.4f}"
)

print(
    f"Test R²  : "
    f"{comparison.iloc[0]['Test_R2']:.4f}"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison.to_csv(
    RESULT_FILE,
    index=False
)

print("\nComparison saved to:")
print(RESULT_FILE)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)