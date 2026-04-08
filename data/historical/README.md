# Historical Data Lane - LOCKED FOR MODEL DEVELOPMENT ONLY

This directory contains the immutable training corpus used for all model development, training, and backtesting.

## Policy

- **Read-Only for Training**: All ML pipelines and training workflows read ONLY from this lane.
- **Never Overwrite**: Files in this lane are never modified or deleted to preserve reproducibility.
- **Versioned by Date**: Each snapshot is named with its date range to enable traceability.

## Directory Structure

- `raw/`: Original ingested data split by source
  - `demand_load/` - load-only source files
  - `weather/` - weather-only source files
  - `calendar/` - holiday/calendar reference files
  - `electricity_demand_2021-01-01_to_2026-04-06.csv` - Combined merged dataset at the root of `raw/` with `is_holiday`
  - `ingestion_manifest_2021-01-01_to_2026-04-06.json` - Ingestion metadata

- `feature_engineered/`: Feature-only outputs from the raw combined file, no splitting
  - `featured_data.parquet` - Raw data after feature engineering

- `final_processed/`: Cleaning, imputation, and final split outputs used for training
  - `cleaned_data.parquet` - Feature-engineered data after cleaning/imputation
  - `train_data.parquet`, `val_data.parquet`, `test_data.parquet` - Chronological splits
  - `prep_metadata.json` - Data quality and preprocessing report
  - `data_quality_report.json` - Stage-wise quality report

## SRS Alignment

- **NFR-SAFE-1**: Ensures historical load and weather data is not lost or corrupted during incremental updates.
- **NFR-SAFE-2**: Safe update mechanism prevents overwriting during operational scraping.
- **NFR-SEC-2**: Local data storage with clear separation from operational ephemeral data.
- **D-DB-01**: Database schema isolation for training vs. operational queries.

## Future Workflows

When retraining on expanded data:
1. Create a new dated snapshot in `operational/raw/` after scraping.
2. Explicitly promote it to a new historical version via an admin command.
3. Never merge operational into historical unintentionally.
4. Archive or version old historical snapshots if needed.
