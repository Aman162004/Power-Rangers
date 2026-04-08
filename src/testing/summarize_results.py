from pathlib import Path

import pandas as pd


csv_path = Path("models/testing/20260408_070201_epoch1_test/test_predictions_vs_actual.csv")
df = pd.read_csv(csv_path)

print("rows", len(df))
print("actual_mean", round(df["actual_load_mw"].mean(), 2))
print("pred_mean", round(df["p50"].mean(), 2))
print("actual_minmax", round(df["actual_load_mw"].min(), 2), round(df["actual_load_mw"].max(), 2))
print("pred_minmax", round(df["p50"].min(), 2), round(df["p50"].max(), 2))
print("max_abs_error", round(df["abs_error_p50"].max(), 2))
print(df.head(3).to_string(index=False))
print(df.tail(3).to_string(index=False))
