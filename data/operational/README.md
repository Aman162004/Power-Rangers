# Operational Data Lane - EPHEMERAL FUTURE SCRAPING

This directory is reserved for future daily scraping and operational inference data.

## Policy

- **Ephemeral**: Data here is temporary and subject to cleanup or refresh.
- **Never Used for Training**: Training pipelines NEVER read from this lane.
- **Isolated from Historical**: Operational data is kept completely separate to prevent contamination of training corpus.
- **For Inference Only**: Once the model is trained, new operational data is used only for inference and forecasting.

## Directory Structure

- `raw/`: Future daily/weekly scraping outputs
  - `load_sldc_{start_date}_to_{end_date}.csv` - Recent SLDC load snapshots
  - `weather_openmeteo_{start_date}_to_{end_date}.csv` - Recent weather snapshots
  - `electricity_demand_{start_date}_to_{end_date}.csv` - Recent merged data
  - `ingestion_manifest_{start_date}_to_{end_date}.json` - Manifest for each scrape run

- `processed/`: Feature-engineered operational data (if needed for inference automation)
  - Processed versions of raw operational data for model inference

## Expected Workflow

1. **Daily/Weekly Scraping**: Incoming scrape writes to `raw/` with dated filenames.
2. **Inference Preprocessing**: Operational data is preprocessed here for model inference.
3. **Historical Snapshot Creation** (Optional): When building next training corpus, explicitly promote selected operational snapshots to historical lane.
4. **Cleanup**: Old operational data can be archived or deleted with no impact on training reproducibility.

## SRS Alignment

- **NFR-SAFE-1 & NFR-SAFE-2**: Operational and historical lanes are completely isolated, making incremental updates safe.
- **NFR-SEC-2**: Separation of concerns ensures local data storage policies are maintained.
- **Graceful Degradation**: If operational scraping fails, historical models continue to serve forecasts.

## Future Expansion

This lane supports:
- Real-time forecast generation on live data.
- Scenario simulation with adjusted weather/load.
- Model performance monitoring against live observations.
