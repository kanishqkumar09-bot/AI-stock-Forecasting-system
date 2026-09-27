from pathlib import Path
import pandas as pd


print("=" * 70)
print("AI FINANCIAL INTELLIGENCE")
print("STOCK MODEL COMPARISON")
print("=" * 70)

# Model performance from our stock-model experiments
results = [
    {
        "Model": "Baseline",
        "MAE": 36.1782,
        "RMSE": 51.2832,
        "MAPE": 1.46
    },
    {
        "Model": "Linear Regression",
        "MAE": 36.0109,
        "RMSE": 51.0807,
        "MAPE": 1.40
    },
    {
        "Model": "Random Forest",
        "MAE": 114.9984,
        "RMSE": 138.8094,
        "MAPE": 4.75
    },
    {
        "Model": "XGBoost",
        "MAE": 86.7333,
        "RMSE": 106.8490,
        "MAPE": 3.54
    }
]

df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE")
print("-" * 70)
print(df.to_string(index=False))

# Select best model using lowest MAE
best_model = df.loc[df["MAE"].idxmin()]

print("\n" + "=" * 70)
print("BEST STOCK MODEL")
print("=" * 70)

print(f"Model : {best_model['Model']}")
print(f"MAE   : {best_model['MAE']:.4f}")
print(f"RMSE  : {best_model['RMSE']:.4f}")
print(f"MAPE  : {best_model['MAPE']:.2f}%")

print("\nReason:")
print("Lower MAE means smaller average prediction error.")

# Save comparison
project_root = Path(__file__).resolve().parents[2]

output_dir = project_root / "data" / "processed" / "stock"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "stock_model_comparison.csv"

df.to_csv(output_file, index=False)

print("\nComparison saved to:")
print(output_file)

print("\n" + "=" * 70)
print("STOCK MODEL COMPARISON COMPLETE")
print("=" * 70)