"""
add_lag672.py — One-shot script to add load_lag_672 (7-day lag) to the
existing train/val/test parquet files without re-running the full pipeline.

Why this is safe:
- Lags are deterministic: load_lag_672[i] = load_mw[i-672]. No randomness.
- We concatenate all splits (time-ordered) before shifting so that the lag
  is computed correctly across split boundaries (e.g. val row 0 gets its lag
  from train row -672, not NaN).
- We then re-split at the original boundary indices.
"""

import pandas as pd
from pathlib import Path

BASE = Path("data/historical/final_processed")
LAG = 672  # 7 days @ 15-min cadence

print("[LAG672] Loading splits...")
train = pd.read_parquet(BASE / "train_data.parquet")
val   = pd.read_parquet(BASE / "val_data.parquet")
test  = pd.read_parquet(BASE / "test_data.parquet")

n_train = len(train)
n_val   = len(val)
n_test  = len(test)
print(f"  train={n_train}, val={n_val}, test={n_test}")

# Concatenate in time order so lag crosses split boundaries correctly
full = pd.concat([train, val, test], ignore_index=True)
full = full.sort_values("time_idx").reset_index(drop=True)

# Compute lag_672 on the full series
print(f"[LAG672] Computing load_lag_{LAG}...")
full[f"load_lag_{LAG}"]              = full["load_mw"].shift(LAG)
full[f"load_lag_{LAG}_was_missing"]  = full[f"load_lag_{LAG}"].isna().astype(int)

# Fill NaN at the very start with forward fill → backfill strategy
# (only affects first 672 rows of the training set — about 7 days)
full[f"load_lag_{LAG}"] = (
    full[f"load_lag_{LAG}"]
    .fillna(method="bfill")   # use nearest known future if head is NaN
    .fillna(method="ffill")   # fallback: shouldn't be needed
)

print(f"  Remaining NaNs: {full[f'load_lag_{LAG}'].isna().sum()}")

# Re-split at original boundaries
train_new = full.iloc[:n_train].copy()
val_new   = full.iloc[n_train:n_train + n_val].copy()
test_new  = full.iloc[n_train + n_val:].copy()

# Verify sizes unchanged
assert len(train_new) == n_train, "train size mismatch!"
assert len(val_new)   == n_val,   "val size mismatch!"
assert len(test_new)  == n_test,  "test size mismatch!"

# Save back
print("[LAG672] Saving updated parquets...")
train_new.to_parquet(BASE / "train_data.parquet", index=False)
val_new.to_parquet(BASE   / "val_data.parquet",   index=False)
test_new.to_parquet(BASE  / "test_data.parquet",  index=False)

# Verify
df_check = pd.read_parquet(BASE / "train_data.parquet")
lag_cols = [c for c in df_check.columns if "lag" in c or "rolling" in c]
print(f"[LAG672] Done! Lag/rolling columns now in parquets: {lag_cols}")
