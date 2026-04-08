# Feature Engineered Lane

This folder contains the first processing output from the raw combined file.

## Contents

- `featured_data.parquet`: feature-engineered data only

## Policy

- No train/validation/test splitting happens here.
- This is the output of the feature engineering stage only.
- Cleaning, imputation, and final splitting belong in `final_processed/`.