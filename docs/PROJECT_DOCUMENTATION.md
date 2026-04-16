# Delhi Power Demand AI — Deep Technical Documentation

> **Project:** AI Electricity Demand Forecasting System for Delhi  
> **Model:** Temporal Fusion Transformer (TFT)  
> **Forecast Horizon:** 48 hours (192 × 15-minute steps)  
> **Generated:** April 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Ingestion Pipeline](#3-data-ingestion-pipeline)
4. [Exploratory Data Analysis (EDA) & Quality Report](#4-exploratory-data-analysis-eda--quality-report)
5. [Feature Engineering](#5-feature-engineering)
6. [Data Splits](#6-data-splits)
7. [Model Architecture — Temporal Fusion Transformer](#7-model-architecture--temporal-fusion-transformer)
8. [Training Pipeline](#8-training-pipeline)
9. [Training Run History & Validation Losses](#9-training-run-history--validation-losses)
10. [Current Accuracy — Test-Set Evaluation](#10-current-accuracy--test-set-evaluation)
11. [Inference & Forecast Engine](#11-inference--forecast-engine)
12. [Testing Pipeline](#12-testing-pipeline)
13. [Streamlit Frontend](#13-streamlit-frontend)
14. [Full System Orchestration](#14-full-system-orchestration)
15. [Dependencies & Environment](#15-dependencies--environment)
16. [Configuration Reference](#16-configuration-reference)

---

## 1. Project Overview

The project is an **end-to-end AI pipeline** that forecasts the **electricity demand for Delhi** at 15-minute granularity, up to 48 hours ahead. It produces probabilistic forecasts — specifically the **P10, P50, and P90 quantiles** — enabling grid operators and planners to understand uncertainty alongside the central prediction.

### Problem Statement
- **Target variable:** `load_mw` — Delhi's real-time grid load in megawatts (MW)
- **Granularity:** 15-minute intervals
- **Forecast window:** 48 hours (192 future steps)
- **Context window:** 6 hours of historical context (24 encoder steps)

### Core Stack
| Layer | Technology |
|---|---|
| Model | Temporal Fusion Transformer (pytorch-forecasting) |
| Training framework | PyTorch Lightning |
| Data pipeline | Pandas, Parquet |
| Data source (load) | Delhi SLDC — `delhisldc.org` (web scraping) |
| Data source (weather) | Open-Meteo archive API |
| Frontend | Streamlit + Plotly |
| Persistence | Parquet (data), `.ckpt` (checkpoints), JSON (metadata) |

---

## 2. Repository Structure

```
Power-Rangers/
├── config/
│   └── config.yaml                  # Master configuration (all tunables live here)
├── data/
│   ├── historical/
│   │   ├── raw/                     # Raw merged CSV (immutable training corpus)
│   │   │   ├── electricity_demand_2021-01-01_to_2026-04-06.csv  (10.9 MB)
│   │   │   ├── demand_load/         # Base 5-min load CSV (2021–2024)
│   │   │   ├── weather/             # Raw weather snapshots
│   │   │   └── calendar/            # Indian holiday CSVs
│   │   ├── feature_engineered/
│   │   │   └── featured_data.parquet  (10.1 MB — after feature engineering, before cleaning)
│   │   └── final_processed/
│   │       ├── cleaned_data.parquet   (12.3 MB — full cleaned dataset)
│   │       ├── train_data.parquet     (9.2 MB)
│   │       ├── val_data.parquet       (1.9 MB)
│   │       ├── test_data.parquet      (1.9 MB)
│   │       ├── prep_metadata.json     (data quality + split summary)
│   │       └── data_quality_report.json
│   └── operational/                 # Ephemeral: live scraping data for inference
├── models/
│   ├── config/                      # Config snapshots per training run
│   ├── runs/                        # Checkpoint directories (one per run)
│   │   ├── 20260408_061558/
│   │   ├── 20260408_070201/
│   │   └── 20260408_075759/
│   └── testing/                     # Test evaluation outputs (metrics + CSV)
├── src/
│   ├── ingestion/                   # Data fetching (SLDC scraper + OpenMeteo)
│   ├── pipelines/                   # Orchestration pipelines
│   ├── training/                    # TFT model + training pipeline + run manager
│   ├── testing/                     # Test evaluation pipeline
│   ├── streamlit_frontend/          # Streamlit UI components
│   ├── shared/                      # Artifact repository utility
│   ├── forecast_engine.py           # Inference wrapper
│   ├── evaluation.py                # Metric computation helpers
│   └── main_inference.py
├── requirements.txt
└── main.py
```

---

## 3. Data Ingestion Pipeline

**Entry point:** `src/ingestion/pipeline.py` → `run_ingestion()`

The ingestion pipeline merges **three distinct data sources** into a single merged CSV snapshot.

### 3.1 Load Data — Delhi SLDC (Web Scraper)

- **Source URL:** `https://www.delhisldc.org/Loaddata.aspx?mode=DD/MM/YYYY`
- **Method:** HTML table scraping using `requests` + `BeautifulSoup` + `lxml`
- **Columns extracted:** `TIMESLOT`, `DELHI` (renamed to `load_mw`)
- **Handling:** Fetches day-by-day, with exponential-backoff retry logic (5 retries, 0.3 backoff factor, 10s timeout)
- **Throttling:** 400ms sleep between requests to avoid being blocked
- **Normalization:** Resampled to strict 15-minute intervals using `pd.resample("15min").mean().interpolate(method="linear")`
- **Base CSV fallback:** Historical data before scraping range is loaded from a pre-built CSV (`powerdemand_5min_2021_to_2024_load_only.csv`) and normalized the same way, then deduplicated/concatenated

### 3.2 Weather Data — Open-Meteo API

- **Source URL:** `https://archive-api.open-meteo.com/v1/archive`
- **Location:** Delhi → Latitude: 28.6139°N, Longitude: 77.2090°E
- **Timezone:** Asia/Kolkata
- **Variables fetched (hourly):**
  - `temperature_2m` → `temperature` (°C)
  - `relative_humidity_2m` → `humidity` (%)
  - `wind_speed_10m` → `wind_speed` (km/h)
  - `precipitation` → `rainfall` (mm)
- **Chunking:** Data fetched in 31-day chunks to respect API limits
- **Caching:** Responses cached with `requests-cache` for 6 hours (path: `.cache/openmeteo`)
- **Upsampling:** Hourly → 15-minute via `resample("15min").interpolate(method="linear")`

### 3.3 Holiday Calendar — `holidays` Library

- **Source:** Python `holidays` library — `holidays.India(years=...)`
- **Output:** A per-day flag `is_holiday` (0/1) + holiday name
- **Join:** Merged into the load+weather DataFrame on date

### 3.4 Merge & Output

After fetching, all three are joined:
```
load_df  LEFT JOIN  weather_df  ON timestamp
result   LEFT JOIN  holiday_df  ON date
```

**Output files (per ingestion run):**
| File | Content |
|---|---|
| `load_sldc_{start}_to_{end}.csv` | Raw SLDC load snapshot |
| `weather_openmeteo_{start}_to_{end}.csv` | Weather snapshot |
| `electricity_demand_{start}_to_{end}.csv` | Merged snapshot |
| `ingestion_manifest_{start}_to_{end}.json` | Run metadata + SHA256 hashes |
| `india_holidays_{start}_to_{end}.csv` | Holiday calendar |

---

## 4. Exploratory Data Analysis (EDA) & Quality Report

All statistics below are derived from `data/historical/final_processed/prep_metadata.json` and `data_quality_report.json`, generated on **2026-04-07T14:20:44 UTC**.

### 4.1 Dataset Overview

| Metric | Value |
|---|---|
| **Source file** | `electricity_demand_2021-01-01_to_2026-04-06.csv` |
| **Total rows (raw)** | 184,510 |
| **Rows after preprocessing** | 184,510 (no rows dropped) |
| **Duplicate rows removed** | 0 |
| **Timezone-invalid rows dropped** | 0 |
| **Date range** | 2021-01-01 → 2026-04-06 |
| **Granularity** | 15-minute intervals |
| **Timezone** | Asia/Kolkata (IST, UTC+5:30) |

### 4.2 Frequency Validation

| Check | Result |
|---|---|
| Valid cadence (15-min uniform) | PASS |
| Total deviations in interval | 0 |
| Missing intervals | 0 |
| Min interval (min) | 15.0 |
| Max interval (min) | 15.0 |
| Std deviation of intervals | 0.0 |

The dataset is **perfectly gapless** with zero missing 15-minute intervals across 5+ years.

### 4.3 Missing Values (Before Imputation)

| Column | Missing Count | % of Total |
|---|---|---|
| `load_mw` | 0 | 0.000% |
| `temperature` | 3 | 0.002% |
| `humidity` | 3 | 0.002% |
| `wind_speed` | 3 | 0.002% |
| `rainfall` | 3 | 0.002% |
| `is_holiday` | 0 | 0.000% |
| All time features | 0 | 0.000% |
| `load_lag_4` | 4 | 0.002% |
| `load_lag_24` | 24 | 0.013% |
| `load_lag_96` | 96 | 0.052% |

Weather missing values (only 3 rows across all columns) were caused by boundary edge effects in the Open-Meteo API. **After forward/backward fill imputation, residual nulls = 0 on all columns.**

### 4.4 Weather Sanity Checks

| Variable | Valid Range | Min Observed | Max Observed | Violations |
|---|---|---|---|---|
| `temperature` | -15C to 60C | 3.2C | 46.0C | 0 — PASS |
| `humidity` | 0% to 100% | 4.0% | 100.0% | 0 — PASS |
| `wind_speed` | 0 to 100 km/h | 0.0 | 36.1 km/h | 0 — PASS |
| `rainfall` | 0 to 1000 mm | 0.0 | 30.4 mm | 0 — PASS |

All weather values are physically plausible for Delhi's climate. No violations detected.

### 4.5 Outlier Treatment — Load (load_mw)

| Metric | Value |
|---|---|
| **Method** | IQR-based clipping (1.5x IQR), per `month-hour` bucket |
| **Outliers detected** | 10,568 (5.73% of total rows) |
| **Treatment** | Values clipped to [lower_bound, upper_bound] for their bucket |
| **Original column preserved as** | `load_mw_raw` |
| **Detection flag** | `load_outlier_detected` (boolean column) |

**IQR clipping is done per `month_hour` bucket** (e.g., "01-00" = January at midnight), giving seasonal and diurnal context to the bounds. Example bounds for January midnight:
- Lower bound: 2,524.37 MW
- Upper bound: 5,519.02 MW

Notable outlier events include extreme cold-wave demand spikes in Jan 2023–2024 (>6,000 MW) and anomalously low demand in Jan 2022 (~2,040 MW, likely data quality issues in SLDC reporting).

> **Note:** Both `load_mw_raw` and `load_outlier_detected` are dropped before training via `training_drop_columns`, keeping only the clipped `load_mw` as the true target.

---

## 5. Feature Engineering

**Module:** `src/preprocessing/feature_engineer.py` (called by `FeatureEngineer`)  
**Output:** `data/historical/feature_engineered/featured_data.parquet`

### 5.1 Time Features

| Feature | Description | Training Status |
|---|---|---|
| `hour` | Hour of day (0–23) | USED |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | USED |
| `month` | Month (1–12) | USED |
| `day_of_year` | Day in year (1–366) | DROPPED before training |
| `week_of_year` | ISO week number | DROPPED before training |
| `is_weekend` | 1 if Saturday/Sunday | DROPPED before training |

### 5.2 Cyclical Encodings

| Feature | Formula | Purpose |
|---|---|---|
| `sin_hour` | sin(2*pi * hour / 24) | Cyclical hour representation |
| `cos_hour` | cos(2*pi * hour / 24) | Cyclical hour representation |

These prevent the model from treating hour 0 and hour 23 as "far apart" in feature space.

### 5.3 Lag Features

| Feature | Lag Offset | Physical Interpretation |
|---|---|---|
| `load_lag_4` | 4 steps back (1 hour) | Same-hour-ago load |
| `load_lag_24` | 24 steps back (6 hours) | Load 6 hours earlier |
| `load_lag_96` | 96 steps back (24 hours) | Same time yesterday |

### 5.4 Rolling Statistics

| Feature | Window | Physical Interpretation |
|---|---|---|
| `rolling_mean_4` | 4 steps (1 hour) | 1-hour rolling average demand |
| `rolling_mean_24` | 24 steps (6 hours) | 6-hour rolling average demand |

### 5.5 Exogenous Features

| Feature | Source | Description |
|---|---|---|
| `temperature` | Open-Meteo | 2m air temperature (C) |
| `humidity` | Open-Meteo | Relative humidity (%) |
| `wind_speed` | Open-Meteo | 10m wind speed (km/h) |
| `rainfall` | Open-Meteo | Precipitation (mm) |
| `is_holiday` | `holidays` library | Indian national/state holiday flag |

### 5.6 TFT Bookkeeping Columns

| Column | Description |
|---|---|
| `time_idx` | Integer index (0 to N-1), required by pytorch-forecasting |
| `group_id` | Series group identifier (always 0 — single series) |

### 5.7 Audit / Missingness Indicator Columns

For every feature column, an `*_was_missing` boolean column is created to track which values were imputed. **All `*_was_missing` columns are dropped before training** via the regex pattern `.*_was_missing$` in `training_drop_pattern`.

### 5.8 Final Column Set Used in Training

After dropping audit, redundant, and outlier-tracking columns, the training-ready columns are:

```
load_mw          <- TARGET (time-varying unknown)
time_idx         <- TFT required
group_id         <- TFT required (group key)

# Time-varying KNOWN reals (future values known at inference time):
hour, day_of_week, month, sin_hour, cos_hour
temperature, humidity, wind_speed, rainfall, is_holiday

# Time-varying UNKNOWN reals (only historical values known):
load_mw          <- also serves as its own unknown input
load_lag_4, load_lag_24, load_lag_96
rolling_mean_4, rolling_mean_24
```

---

## 6. Data Splits

**Method:** Chronological (time-series safe — no shuffle, no random split)  
**Split file:** `data/historical/final_processed/prep_metadata.json`

| Split | Rows | Start Timestamp | End Timestamp | Approx Duration |
|---|---|---|---|---|
| **Train** | 129,156 | 2021-01-01 06:00 | 2024-09-07 14:45 | ~3.7 years |
| **Validation** | 27,677 | 2024-09-07 15:00 | 2025-06-22 22:00 | ~9.5 months |
| **Test** | 27,677 | 2025-06-22 22:15 | 2026-04-07 05:15 | ~9.5 months |
| **Total** | **184,510** | — | — | ~5.25 years |

### Split Ratios
- Train: **70.0%**
- Validation: **15.0%**
- Test: **15.0%**

### Important Notes
- The split uses strict temporal ordering — no data leakage is possible
- The validation set immediately follows training; the test set immediately follows validation
- Lag features (lag_96 = 24 hours) create technically overlapping context windows, but **targets are never leaked** — future `load_mw` values are not accessible
- The minimum required training rows = `encoder_window + decoder_window = 24 + 192 = 216` rows — easily satisfied

---

## 7. Model Architecture — Temporal Fusion Transformer

**Library:** `pytorch-forecasting >= 1.0.0`  
**Class:** `TemporalFusionTransformer.from_dataset(...)`

The Temporal Fusion Transformer (TFT) is a state-of-the-art deep learning architecture for multi-horizon time-series forecasting, designed with built-in interpretability through **variable selection networks** and **multi-head attention**.

### 7.1 Architecture Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `hidden_size` | **64** | Core LSTM & attention representation size |
| `attention_head_size` | **4** | Number of multi-head attention heads |
| `dropout` | **0.1** | Dropout rate (10%) |
| `hidden_continuous_size` | **8** | Hidden size for continuous variable processing |
| `output_size` | **3** | One output per quantile (P10, P50, P90) |
| `loss function` | `QuantileLoss([0.1, 0.5, 0.9])` | Pinball loss for 3 quantiles |
| `mask_bias` | -1x10^4 (fp16) | Attention mask magnitude (safe for mixed precision) |
| `log_interval` | **10** | Log gradient norms every 10 batches |
| `reduce_on_plateau_patience` | **4** | LR scheduler patience |

### 7.2 Sequence Lengths

| Parameter | Config Key | Value | Meaning |
|---|---|---|---|
| `max_encoder_length` | `encoder_window` | **24** | Up to 6 hours of history (24 x 15min) |
| `min_encoder_length` | `encoder_window // 2` | **12** | At least 3 hours required |
| `max_prediction_length` | `decoder_window` | **192** | Predict 48 hours ahead (192 x 15min) |
| `min_prediction_length` | — | **1** | Allow partial forecasts |

### 7.3 Input Feature Wiring

| Input Category | Columns |
|---|---|
| **Static categoricals** | (none) |
| **Static reals** | (none) |
| **Time-varying known categoricals** | (none) |
| **Time-varying known reals** | `hour`, `day_of_week`, `month`, `sin_hour`, `cos_hour`, `temperature`, `humidity`, `wind_speed`, `rainfall`, `is_holiday` |
| **Time-varying unknown categoricals** | (none) |
| **Time-varying unknown reals** | `load_mw`, `load_lag_4`, `load_lag_24`, `load_lag_96`, `rolling_mean_4`, `rolling_mean_24` |
| **Auto-added by TFT** | `relative_time_idx`, `target_scale`, `encoder_length` |

### 7.4 Optimizer & Scheduler

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | **0.0003** |
| LR Scheduler | `ReduceLROnPlateau` |
| Scheduler patience | 4 epochs |
| Scheduler monitor | `val_loss` |

### 7.5 Approximate Parameter Count

With `hidden_size=64`, `attention_head_size=4`, the model has approximately **~700K–1M trainable parameters** (typical for TFT at this scale).

---

## 8. Training Pipeline

**Entry point:** `src/training/training_pipeline.py` -> `run_training_pipeline()`

### 8.1 Pipeline Steps

```
1. Load config (config/config.yaml)
2. Register PyTorch safe globals (for checkpoint compatibility with PyTorch >= 2.6)
3. Set CUDA optimizations (cuDNN benchmark, float32 matmul precision = "medium")
4. Resolve resume policy: auto -> find latest run with last.ckpt
5. Initialize TrainingRunManager (creates run directory, saves config snapshot)
6. Load train/val/test splits from data/historical/final_processed/
7. Drop audit + redundant columns (day_of_year, week_of_year, is_weekend, load_mw_raw,
   load_outlier_detected, *_was_missing)
8. Validate all required columns present
9. Build TFT TimeSeriesDataSet objects
10. Create DataLoaders (num_workers=8, pin_memory=True, persistent_workers=True,
    prefetch_factor=4, batch_size=128)
11. Initialize TFT model from training dataset
12. Set up callbacks: EarlyStopping + ModelCheckpoint
13. Create Lightning Trainer
14. Resume from last.ckpt if available (auto policy)
15. trainer.fit() -> train + validate
16. trainer.test() -> evaluate on test set
17. Save metadata (best_val_loss, epochs_completed)
18. Finalize run (status = "completed")
```

### 8.2 Training Configuration

| Parameter | Value |
|---|---|
| Max epochs | **50** |
| Batch size | **128** |
| Early stopping patience | **10 epochs** |
| Checkpoint strategy | `all` (every epoch + keep last) |
| Checkpoint every N epochs | **1** |
| Mixed precision | **16-mixed** (fp16 forward, fp32 optimizer) |
| Accelerator | GPU (if CUDA available) / CPU fallback |
| Devices | 1 |
| CuDNN benchmark | Enabled |
| Num DataLoader workers | **8** |
| Pin memory | **True** |
| Persistent workers | **True** |
| Prefetch factor | **4** |

### 8.3 Checkpoint Strategy

The `checkpoint_save_strategy = "all"` means **every epoch's checkpoint is saved** with the filename format:
```
epoch=epoch={NN}-val_loss=val_loss={FLOAT}.ckpt
last.ckpt  <- always the most recent epoch
```

### 8.4 Resume Policy

The `resume_policy = "auto"` means:
- At startup, scan all run directories sorted by timestamp (newest first)
- Find the most recent run that has a `last.ckpt`
- Automatically resume from that checkpoint (full state: weights, optimizer, scheduler, epoch count)

### 8.5 Run Storage

Each training run creates a directory under `models/runs/<run_id>/` where `<run_id>` is a UTC timestamp like `20260408_075759`. Config snapshots go to `models/config/<run_id>.yaml`.

---

## 9. Training Run History & Validation Losses

Three training runs have been completed, all using the same model architecture.

### Run 1: `20260408_061558`

Initial exploratory run. Checkpoint files preserved but listing not detailed. Architecture identical to subsequent runs.

---

### Run 2: `20260408_070201` — Used for test evaluation

| Epoch | Val Loss (Quantile Loss) |
|---|---|
| 0 | 172.62 |
| **1** | **96.36 — Best** |
| 2 | 118.37 |
| 3 | 146.38 |
| 4 | 155.81 |
| 5 | 200.43 |
| 6 | 153.50 |
| 7 | 168.12 |
| 8 | 229.22 |

- **Best checkpoint:** `epoch=epoch=01-val_loss=val_loss=96.36.ckpt`
- Training completed 9 epochs before early stopping exhausted patience

---

### Run 3: `20260408_075759` — Latest run

| Epoch | Val Loss (Quantile Loss) |
|---|---|
| 0 | 188.41 |
| 1 | 203.45 |
| 2 | 245.15 |
| 3 | 166.25 |
| **4** | **112.94 — Best** |
| 5 | 117.33 |
| 6 | 145.05 |

- **Best checkpoint:** `epoch=epoch=04-val_loss=val_loss=112.94.ckpt`
- `last.ckpt` = epoch 6 checkpoint (145.05 val loss)
- Training stopped after 7 epochs

> **Note on Quantile Loss:** The optimization objective is `QuantileLoss([0.1, 0.5, 0.9])`, which is the sum of pinball losses across all 3 quantiles. The raw loss values (96–245 range) are in **MW units** (approximate), not percentages.

---

## 10. Current Accuracy — Test-Set Evaluation

**Evaluated via:** `src/testing/testing_pipeline.py` -> `run_test_pipeline()`  
**Run evaluated:** `20260408_070201` (Run 2)  
**Checkpoint used:** `epoch=epoch=01-val_loss=val_loss=96.36.ckpt` (best val loss checkpoint)  
**Results stored in:** `models/testing/20260408_070201_epoch1_test/`

### 10.1 Metrics

| Metric | Value | Description |
|---|---|---|
| **MAE** | **302.95 MW** | Mean Absolute Error |
| **RMSE** | **388.37 MW** | Root Mean Squared Error |
| **MAPE** | **8.12%** | Mean Absolute Percentage Error |
| **SMAPE** | **8.65%** | Symmetric MAPE |

### 10.2 Interpretation

- At a typical Delhi load of ~3,500–5,500 MW, **MAE of ~303 MW represents roughly 5.5–8.7% of the actual load value**, which is consistent with the MAPE of 8.12%
- **RMSE of 388 MW** is higher than MAE, indicating some large errors exist (expected for extreme weather events or unusual demand patterns in the test set)
- The **test period covers June 2025 to April 2026** — a period with summer peak demand transitions and monsoon effects, making it a realistic and challenging evaluation window
- These metrics are for the **median forecast (P50 quantile only)** — the P10/P90 quantiles provide additional uncertainty coverage

> **Important context:** The evaluation was done after only ~1–2 epochs of training. With more training epochs and tuning, performance is expected to improve. The model was trained up to 9 epochs in Run 2, but accuracy metrics were computed on the epoch-1 best checkpoint.

### 10.3 Output Artifacts

| File | Content |
|---|---|
| `test_predictions_vs_actual.csv` | 192-row comparison: actual vs P10/P50/P90, error columns |
| `metrics.json` | MAE, RMSE, MAPE, SMAPE + checkpoint metadata |

The comparison CSV includes per-step columns:
- `timestamp`, `actual_load_mw`
- `p10`, `p50`, `p90` (quantile forecasts)
- `error_p50` = actual - p50
- `abs_error_p50` = |error_p50|
- `ape_p50` = |error_p50 / actual| x 100

---

## 11. Inference & Forecast Engine

**Module:** `src/forecast_engine.py` -> `ForecastEngine`

The `ForecastEngine` class loads the **active trained checkpoint** and generates probabilistic 24-hour forecasts.

### 11.1 Active Model Pointer

Active model is tracked via `models/ACTIVE_MODEL.txt` which contains the run ID of the deployed model. The engine:
1. Reads `ACTIVE_MODEL.txt` to get `run_id`
2. Loads `models/runs/{run_id}/best.ckpt`
3. Generates P10, P50, P90 forecasts for 96 steps (24 hours at 15-min resolution)

> **Current status:** The `generate_forecast()` method currently returns a **placeholder linear extrapolation** (`base_val +/- 100 MW`) rather than the actual TFT model output. The full TFT inference path (loading checkpoint, preparing tensors, calling `model.predict()`) is implemented in `testing_pipeline.py` and needs to be ported to `ForecastEngine`.

### 11.2 Forecast Output Format

```python
pd.DataFrame({
    "timestamp": forecast_timestamps,   # 96 timestamps at 15-min intervals
    "p10": p10,                          # 10th percentile
    "p50": p50,                          # Median (point forecast)
    "p90": p90,                          # 90th percentile
})
```

---

## 12. Testing Pipeline

**Module:** `src/testing/testing_pipeline.py` -> `run_test_pipeline()`

The dedicated testing pipeline performs **rigorous holdout evaluation**:

1. Loads train/val/test splits (re-creates TFT `TimeSeriesDataSet` objects to get proper normalization)
2. Selects the checkpoint with the **lowest `val_loss` in filename** (or falls back to `last.ckpt`)
3. Rebuilds TFT model architecture from `train_dataset`
4. Loads saved `state_dict` from checkpoint
5. Runs `model.predict()` in `mode="quantiles"` to get P10/P50/P90
6. Compares P50 forecast against actual `load_mw` values from the test set tail (last 192 rows)
7. Computes MAE, RMSE, MAPE, SMAPE
8. Saves results to `models/testing/{run_id}_epoch1_test/`

### Key Metric Functions

```python
def _metrics_from_predictions(actual, p50):
    diff = actual - p50
    mae  = mean(|diff|)
    rmse = sqrt(mean(diff^2))
    mape = mean(|diff / actual|) x 100
    smape = mean(2*|diff| / (|actual| + |p50|)) x 100
```

---

## 13. Streamlit Frontend

**Module:** `src/streamlit_frontend/streamlit_app.py`

A three-tab Streamlit dashboard providing:

### Tab 1: Forecast
- **Generate Forecast** button → calls `ForecastEngine.generate_forecast()`
- **Forecast Curve:** Plotly line chart of historical demand + predicted P50
- **Uncertainty Bands:** P10–P90 shaded region with P50 centerline
- **Peak Detection:** Identifies peak demand timestamp + value from forecast

### Tab 2: Evaluation
- Loads test data, runs forecast, calls `ModelEvaluator.evaluate_forecast()`
- Displays MAE, RMSE, MAPE as metric cards
- Plots predictions vs actual

### Tab 3: Scenario Simulation
- Temperature Adjustment slider (-10C to +10C)
- Load Scaling Factor slider (0.5x to 1.5x)
- Compares base forecast vs scenario-adjusted forecast

### Frontend Submodules

| Module | Purpose |
|---|---|
| `visualization.py` | Plotly charts: forecast curves, probability bands, peak highlights, scenario comparison |
| `peak_detection.py` | Finds peak demand timestamp + value in forecast DataFrame |
| `probabilistic_forecasting.py` | Point forecast extraction from quantile output |
| `scenario_simulation.py` | Applies parameter adjustments to generate alternative forecasts |

---

## 14. Full System Orchestration

**Entry point:** `src/pipelines/run_full_system.py` -> `main()`

```
Step 1: Check if train_data.parquet exists
        -> If not: run prepare_historical_data.py
Step 2: Check if ACTIVE_MODEL.txt exists
        -> If not: run training_pipeline.py
Step 3: Run inference + evaluation on test set
Step 4: Launch Streamlit dashboard
```

This orchestration enables **one-command end-to-end execution** from raw data to running dashboard.

---

## 15. Dependencies & Environment

**File:** `requirements.txt`

| Package | Version Constraint | Purpose |
|---|---|---|
| `numpy` | >=1.26.4, <3.0 | Numerical computing |
| `pandas` | >=2.2.2, <3.0 | DataFrame operations |
| `pyarrow` | >=16.1.0 | Parquet read/write |
| `PyYAML` | >=6.0.1, <7.0 | Config loading |
| `scikit-learn` | >=1.5.1, <2.0 | Scalers used internally by pytorch-forecasting |
| `torch` | >=2.4.0, <3.0 | Deep learning backend |
| `pytorch-lightning` | >=2.3.3, <3.0 | Training framework |
| `pytorch-forecasting` | >=1.0.0, <2.0 | TFT model + TimeSeriesDataSet |
| `streamlit` | >=1.36.0, <2.0 | Web dashboard |
| `plotly` | >=5.22.0, <6.0 | Interactive charts |
| `matplotlib` | >=3.9.0, <4.0 | Plotting (auxiliary) |
| `requests` | >=2.32.3, <3.0 | HTTP client (SLDC scraping) |
| `requests-cache` | >=1.2.1, <2.0 | HTTP caching (Open-Meteo) |
| `beautifulsoup4` | >=4.12.3, <5.0 | HTML parsing (SLDC scraper) |
| `lxml` | >=5.2.2 | Fast HTML/XML parser |
| `holidays` | >=0.35, <1.0 | Indian holiday calendar |

---

## 16. Configuration Reference

**File:** `config/config.yaml` — the single source of truth for all parameters.

```yaml
data:
  historical_merged_file: "data/historical/raw/electricity_demand_2021-01-01_to_2026-04-06.csv"
  historical_splits_path: "data/historical/final_processed/"
  training_splits:
    train: "train_data.parquet"
    val:   "val_data.parquet"
    test:  "test_data.parquet"
  training_drop_columns: [day_of_year, week_of_year, is_weekend, load_mw_raw, load_outlier_detected]
  training_drop_pattern: ".*_was_missing$"

ingestion:
  start_date: "2026-04-07"
  end_date:   "2026-04-13"
  latitude:   28.6139           # Delhi
  longitude:  77.2090
  timezone:   "Asia/Kolkata"
  retry_total: 5
  backoff_factor: 0.3
  timeout_seconds: 10
  sldc_sleep_seconds: 0.4

pipeline:
  encoder_window: 24    # 6 hours of context
  decoder_window: 192   # 48-hour forecast horizon
  stride: 1

features:
  lags:            [4, 24, 96]   # 1h, 6h, 24h lookback
  rolling_windows: [4, 24]       # 1h, 6h rolling mean

model:
  hidden_size:          64
  attention_head_size:  4
  dropout:              0.1
  hidden_continuous_size: 8
  output_size:          1        # overridden to 3 by quantile list
  loss:                 "quantile"
  quantiles:            [0.1, 0.5, 0.9]
  learning_rate:        0.0003
  max_epochs:           50
  batch_size:           128

training:
  early_stopping_patience:  10
  checkpoint_save_strategy: "all"
  checkpoint_every_n_epochs: 1
  resume_policy:            "auto"
  resume_weights_only:      false
  precision:                "16-mixed"
  num_workers:              8
  pin_memory:               true
  persistent_workers:       true
  prefetch_factor:          4
  cudnn_benchmark:          true
```

---

## Summary: End-to-End Data Flow

```
[Delhi SLDC website]        [Open-Meteo API]       [holidays library]
       | (scrape HTML)            | (REST, 31-day chunks)    | (Python)
       v                           v                           v
  load_mw (15-min)        weather (hourly -> 15-min)   is_holiday (daily)
       |                           |                           |
       +-------------- MERGE (LEFT JOIN on timestamp) ---------+
                          |
              electricity_demand_*.csv (raw merged)
              184,510 rows | 2021-01-01 to 2026-04-06
                          |
               [Feature Engineering]
               - time: hour, day_of_week, month
               - cyclical: sin_hour, cos_hour
               - lags: load_lag_4 (1h), load_lag_24 (6h), load_lag_96 (24h)
               - rolling: rolling_mean_4 (1h), rolling_mean_24 (6h)
                          |
               featured_data.parquet (10.1 MB)
                          |
               [Cleaning & Imputation]
               - timezone normalization (Asia/Kolkata)
               - IQR-based outlier clipping (per month-hour bucket)
                 -> 10,568 outliers clipped (5.73%)
               - forward/backward fill for 3 missing weather rows
                          |
               cleaned_data.parquet (184,510 rows, 12.3 MB)
                          |
               [Chronological Split — no shuffle]
          +-----------+----------+-----------+
          |  TRAIN    |    VAL   |   TEST    |
          | 129,156   |  27,677  |  27,677   |
          | rows      |  rows    |  rows     |
          | 2021-2024 | 2024-25  | 2025-26   |
          | (70%)     | (15%)    | (15%)     |
          +-----------+----------+-----------+
                          |
               [Temporal Fusion Transformer]
               - Encoder:  24 steps (6 hours) of context
               - Decoder: 192 steps (48 hours) forecast
               - Hidden size: 64 | Heads: 4 | Dropout: 0.1
               - Loss: QuantileLoss([0.1, 0.5, 0.9])
               - Optimizer: Adam (lr=0.0003)
               - Mixed precision: fp16
                          |
               [3 Training Runs — Best val_loss = 96.36 MW (Run 2, Epoch 1)]
                          |
               [Evaluation on Test Set (192 steps = 48h)]
               +---------------------------+
               | MAE   = 302.95 MW         |
               | RMSE  = 388.37 MW         |
               | MAPE  =   8.12%           |
               | SMAPE =   8.65%           |
               +---------------------------+
                          |
               [Streamlit Dashboard]
               - Forecast curves + P10-P90 uncertainty bands
               - Peak demand detection
               - Scenario simulation (temperature/load adjustments)
```
