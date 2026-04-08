# Final Processed Lane

This folder contains the cleaning, imputation, and final split outputs used for model training.

## Contents

- `cleaned_data.parquet`: feature-engineered data after cleaning and imputation
- `train_data.parquet`: chronological training split
- `val_data.parquet`: chronological validation split
- `test_data.parquet`: chronological test split
- `prep_metadata.json`: preprocessing summary and split metadata
- `data_quality_report.json`: stage-wise quality report

## Policy

- This is the last stop before model training.
- Splits are created here after cleaning and imputation.
- Do not treat this as raw input; it is derived training data.