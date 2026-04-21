# Power Rangers Project Workflow Report

Audience: a newly joining team member who needs to understand what the project does, how the files connect, and which files are used at runtime.

Scope: this report describes the current codebase as it exists in this repository. It does not describe planned features unless the code contains a scaffold for them.

## 1. What This Project Is

Power Rangers is a Delhi power demand forecasting project. It has:

- A React/Vite web frontend for login, registration, dashboard copy, and forecast visualization.
- A FastAPI backend for auth endpoints, health checks, and forecast generation.
- A data ingestion layer for Delhi SLDC load data and Open-Meteo weather data.
- A historical data preparation pipeline that creates feature-engineered, cleaned, and split training datasets.
- A Temporal Fusion Transformer forecasting path for operational inference.
- Training, testing, and evaluation scripts for model development.
- SQLite-based authentication with roles: `ADMIN`, `ANALYST`, and `OPERATOR`.

The live web workflow currently depends on two running servers:

- Backend API: `http://localhost:8000`
- Frontend app: `http://localhost:3001`

The project is not deployed in the codebase. Users clone it, install dependencies, seed users, start the backend, then start the frontend.

## 2. High-Level System Map

```text
Browser
  |
  | loads React app
  v
frontend/index.html
  -> frontend/src/main.tsx
  -> frontend/src/App.tsx
  -> Login/Register/Dashboard pages
  |
  | auth HTTP calls
  v
FastAPI /api/auth/*
  -> src/auth/routes.py
  -> src/auth/database.py
  -> src/auth/jwt_handler.py
  -> data/auth.db
  |
  | forecast HTTP call
  v
FastAPI /api/forecast
  -> backend/main.py
  -> src/ingestion/load_fetcher.py
  -> src/forecast/tft_inference.py
  -> src/ingestion/weather_fetcher.py
  -> models/final model/*.ckpt
  -> data/historical/final_processed/*.parquet
  -> JSON response
  |
  v
Dashboard chart and metrics
```

There are no true recursive function calls in the main application runtime. The system uses nested call chains and repeated loops. The most important repeated call is that `/api/forecast` may call `fetch_sldc_load_data()` once to fetch recent history and again to attach actual values for the forecast horizon when those actuals already exist.

## 3. Local Startup Workflow

### Backend startup

Command:

```powershell
uvicorn backend.main:app --reload --port 8000
```

File order:

- `backend/main.py` is imported by Uvicorn.
- `backend/main.py` imports forecast, ingestion, fallback, and auth modules.
- `src/auth/routes.py` is imported and creates the auth router.
- `src/auth/routes.py` calls `init_db()` during import.
- `src/auth/database.py` creates the SQLite database and auth tables if needed.
- `backend/main.py` creates the `FastAPI` app.
- `backend/main.py` adds CORS middleware.
- `backend/main.py` registers `auth_router`.
- Backend routes become available.

Available backend routes from current code:

- `GET /api/health`
- `POST /api/forecast`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/refresh`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `POST /api/auth/admin/invite`
- `GET /api/auth/admin/users`
- `PUT /api/auth/admin/users/{user_id}/role`
- `DELETE /api/auth/admin/users/{user_id}`

Important note: older documentation says admin routes are under `/api/admin/...`; current code places them under `/api/auth/admin/...`.

### Admin seed workflow

Command:

```powershell
python src/auth/seed_admin.py
```

File order:

- `src/auth/seed_admin.py`
- `src/auth/database.py`
- `src/auth/models.py`
- `data/auth.db`

What happens:

- The auth database is initialized.
- Existing demo users may be reset.
- Demo users are created:
  - `admin` / `changeme123!` / `ADMIN`
  - `analyst_demo` / `analyst123!` / `ANALYST`
  - `operator_demo` / `operator123!` / `OPERATOR`

### Frontend startup

Command:

```powershell
cd frontend
npm install
npm run dev
```

File order:

- `frontend/package.json` defines `npm run dev` as `vite --port 3001`.
- `frontend/vite.config.ts` confirms the dev server port and alias config.
- Browser loads `frontend/index.html`.
- `frontend/index.html` loads `/src/main.tsx`.
- `frontend/src/main.tsx` mounts React into `#root`.
- `frontend/src/App.tsx` creates the routes and auth bootstrap behavior.
- `frontend/src/hooks/useAuth.ts` checks token state.
- `frontend/src/lib/authClient.ts` is used for auth API calls.
- User lands on `/login`, `/register`, or `/`.

Frontend routes from current code:

- `/login`
- `/register`
- `/`
- `*` redirects to `/`

The current `App.tsx` does not define an `/admin` route. `UserMenu.tsx` contains an Admin Panel button that navigates to `/admin`, but that component is not currently wired into `App.tsx` or the dashboard route.

## 4. Real-Time User Workflow

### First-time user path

- User opens `http://localhost:3001`.
- `frontend/src/App.tsx` checks whether an access token exists in local storage.
- If no token exists, the user is routed to `/login`.
- User can either log in or open the registration page.

Registration path:

- `frontend/src/pages/Register.tsx` collects username, email, password, confirm password, full name, and invite token.
- `Register.tsx` calls `useAuth.register()`.
- `frontend/src/hooks/useAuth.ts` sends `POST /api/auth/register` through `authClient`.
- `frontend/src/lib/authClient.ts` sends the request to `http://localhost:8000/api/auth/register` unless `VITE_API_BASE_URL` is set.
- `src/auth/routes.py` validates the invite token.
- `src/auth/database.py` creates the user with role `OPERATOR`.
- `src/auth/jwt_handler.py` creates access and refresh tokens.
- `src/auth/database.py` stores the refresh token.
- Frontend stores tokens in local storage and routes the user to `/`.

Login path:

- `frontend/src/pages/Login.tsx` collects username and password.
- `Login.tsx` calls `useAuth.login()`.
- `useAuth.ts` sends `POST /api/auth/login`.
- `src/auth/routes.py` calls `UserDB.authenticate_user()`.
- `src/auth/database.py` validates the SHA256 password hash.
- `src/auth/jwt_handler.py` creates access and refresh tokens.
- `src/auth/database.py` stores the refresh token.
- Frontend stores tokens in local storage and routes the user to `/`.

Token refresh path:

- `frontend/src/lib/authClient.ts` adds the access token to outgoing auth requests.
- If a request returns `401`, the Axios interceptor calls `POST /api/auth/refresh`.
- `src/auth/routes.py` verifies the refresh token.
- `src/auth/database.py` checks that the refresh token is active and not expired.
- `src/auth/jwt_handler.py` creates a new access token.
- The original request is retried.
- If refresh fails, local tokens are cleared and the user is redirected to `/login`.

Logout path:

- Dashboard calls `useAuth.logout()`.
- `useAuth.ts` sends `POST /api/auth/logout` when a refresh token exists.
- `src/auth/routes.py` revokes the refresh token.
- Frontend clears tokens and user state.
- User is sent back to `/login`.

## 5. Forecast Dashboard Workflow

The dashboard forecast flow starts in the browser and ends with a JSON response from the backend.

### Frontend forecast trigger

File order:

- `frontend/src/App.tsx`
- `frontend/src/components/ProtectedRoute.tsx`
- `frontend/src/pages/Dashboard.tsx`

Runtime flow:

- Authenticated user opens `/`.
- `ProtectedRoute.tsx` allows the page if `isAuthenticated` is true.
- `Dashboard.tsx` initializes dashboard state.
- `Dashboard.tsx` sets the forecast date to tomorrow by default.
- `Dashboard.tsx` sets the temperature scenario slider to `0`.
- `Dashboard.tsx` runs `loadAnalytics()` on mount and whenever forecast date or temperature delta changes.
- `loadAnalytics()` sends:

```text
POST http://localhost:8000/api/forecast?days_to_fetch=3&forecast_date=<selected-date>&temperature_delta_c=<delta>
```

Important note: forecast calls are hardcoded to `http://localhost:8000` in `Dashboard.tsx`. Auth calls use `VITE_API_BASE_URL` or `http://localhost:8000`.

### Backend forecast route

File order:

- `backend/main.py`
- `src/ingestion/load_fetcher.py`
- `src/forecast/tft_inference.py`
- `src/ingestion/weather_fetcher.py`
- `src/forecast/fallback_data.py` only if inference fails and `ENABLE_DUMMY_FALLBACK=true`

Runtime flow inside `POST /api/forecast`:

- `fetch_and_predict()` receives query parameters.
- `days_to_fetch` is validated.
- Forecast date is resolved.
- Aggressiveness is calculated:
  - If `temperature_delta_c` is provided, backend converts it to scenario scaling with `temperature_delta_c * 2`.
  - This value is limited to `-5` through `5`.
  - If direct `aggressiveness_pct` is used instead, it is limited to `-10` through `10`.
- Recent history window is calculated.
- `fetch_sldc_load_data()` pulls recent Delhi SLDC demand history.
- The recent history is saved to `data/operational/recent_load.csv`.
- `_build_forecast_tft()` calls `run_tft_inference()`.
- `_apply_aggressiveness()` adjusts quantile predictions if scenario scaling is non-zero.
- `_attach_actuals_for_horizon()` tries to fetch real actual load for the same forecast horizon when the horizon is not fully in the future.
- `_compute_forecast_metrics()` calculates MAE, RMSE, and MAPE when actuals are available.
- JSON is returned to the frontend.

### SLDC load fetch nested workflow

Main entry:

- `src/ingestion/load_fetcher.py`
- Function: `fetch_sldc_load_data(start_date, end_date, ...)`

Nested call order:

- `fetch_sldc_load_data()`
- `_build_session()`
- `_iter_dates()`
- For each date:
  - `_load_cached_day()`
  - `_scrape_sldc_day()` if cache is missing or stale
  - `_extract_load_table()`
  - `_normalise_day_frame()`
  - `_write_day_cache()`
- Daily frames are concatenated.
- Data is sorted and duplicate timestamps are removed.
- Load is resampled to 15-minute cadence.
- Missing load values are time-interpolated, forward-filled, and backward-filled.
- Final dataframe with `timestamp` and `load_mw` is returned.

Cache behavior:

- Day-level raw cache files are written under `data/operational/raw/`.
- Current-day cache can expire quickly.
- Older cached dates are reused.

### TFT inference nested workflow

Main entry:

- `src/forecast/tft_inference.py`
- Function: `run_tft_inference(load_df=None, forecast_date=None, horizon_steps=None)`

Nested call order:

- `run_tft_inference()`
- `_register_safe_globals()`
- `_load_config()`
- `_find_latest_checkpoint()`
- `_load_tft_model()`
  - `_load_training_splits()`
  - `src/training/training_pipeline.py::_drop_unused_training_columns()`
  - `src/training/training_pipeline.py::_build_tft_datasets()`
  - `TemporalFusionTransformer.from_dataset()`
  - `model.load_state_dict()`
- `_prepare_history()`
- `fetch_sldc_load_data()` if no recent history was passed in
- `fetch_openmeteo_weather()`
- `holidays.India()` for holiday flags
- Future rows are built in a loop for the decoder horizon.
- `TimeSeriesDataSet.from_dataset()` creates prediction data.
- `model.predict(..., mode="quantiles")` creates p10, p50, and p90 predictions.
- Dataframe with quantile predictions is returned.

Forecast output:

- `timestamp`
- `p10`
- `p50`
- `p90`
- `predicted_load_mw`
- `actual_load_mw` after actuals are attached by `backend/main.py`

Important current behavior:

- The model checkpoint search prefers files in `models/final model/`.
- In the final-model directory, files are sorted lexicographically and the last one is used.
- Current final files include val loss `59.31` and `96.36`; the code may select `96.36` because it is lexicographically last, not because it has the best validation loss.

### Weather fetch nested workflow

Main entry:

- `src/ingestion/weather_fetcher.py`
- Function: `fetch_openmeteo_weather(start_date, end_date, latitude, longitude, ...)`

Nested behavior:

- Chooses historical archive API or forecast API based on date range.
- Splits long ranges into chunks.
- Fetches hourly weather.
- Keeps temperature, humidity, wind speed, and precipitation.
- Resamples data to 15-minute cadence.
- Interpolates missing values.
- Returns weather dataframe for feature construction.

### Frontend rendering after forecast response

File order:

- `frontend/src/pages/Dashboard.tsx`
- `frontend/src/components/aceternity/sidebar-demo.tsx`
- `frontend/src/components/ui/sidebar.tsx`
- `frontend/src/components/features/functional-footer.tsx`
- Other visual components used by dashboard sections.

Runtime behavior:

- `Dashboard.tsx` parses backend JSON.
- It stores:
  - `metrics`
  - `peak`
  - `points`
  - `avgTemperatureC`
- `chartData` is recalculated with `useMemo`.
- SVG paths are built for p10, p50, p90, actuals, and uncertainty band.
- User can change the date.
- User can change temperature delta.
- Each date or temperature change refetches forecast data.

## 6. Offline Data Preparation Workflow

Command:

```powershell
python src/pipelines/prepare_historical_data.py
```

File order:

- `src/pipelines/prepare_historical_data.py`
- `config/config.yaml`
- `src/feature_engineer.py`
- `src/dataset_builder.py`
- `src/data_quality.py`
- `src/shared/artifact_repository.py`
- `data/historical/raw/electricity_demand_2021-01-01_to_2026-04-06.csv`
- `data/historical/feature_engineered/featured_data.parquet`
- `data/historical/final_processed/cleaned_data.parquet`
- `data/historical/final_processed/train_data.parquet`
- `data/historical/final_processed/val_data.parquet`
- `data/historical/final_processed/test_data.parquet`
- `data/historical/final_processed/prep_metadata.json`
- `data/historical/final_processed/data_quality_report.json`

Workflow:

- Load project config.
- Read locked historical merged CSV.
- Add time, cyclical, lag, and rolling features.
- Save feature-engineered parquet.
- Clean data, handle timestamps, remove duplicate timestamps, mark missing values, fill missing values.
- Save cleaned parquet.
- Add `time_idx` and `group_id`.
- Chronologically split into train, validation, and test data.
- Save all split parquet files.
- Save metadata and quality report.

Current code note:

- `docs/data/PREPROCESSING_HARDENING.md` describes a richer validation implementation than the current `src/data_quality.py` file exposes. The current code has a simpler `DataQualityReporter`; it does not contain the documented `TimezoneNormalizer`, `FrequencyValidator`, `WeatherSanityChecker`, or `OutlierDetector` classes.

## 7. Optional Ingestion Workflow

Command:

```powershell
python src/pipelines/main_ingestion.py
```

File order:

- `src/pipelines/main_ingestion.py`
- `src/ingestion/pipeline.py`
- `config/config.yaml`
- `src/ingestion/load_fetcher.py`
- `src/ingestion/weather_fetcher.py`
- `data/operational/raw/*.csv`
- `data/operational/raw/*.json`

Workflow:

- Load date range and location from config.
- Optionally normalize the configured base load CSV.
- Scrape missing or requested Delhi SLDC load dates.
- Fetch Open-Meteo weather.
- Generate India holiday/calendar rows.
- Merge load, weather, and holiday data.
- Write operational snapshots and manifest files.

Policy:

- `data/operational/` is for temporary operational inference data.
- Training should use `data/historical/`, not operational files.

## 8. Training Workflow

Command:

```powershell
python src/training/training_pipeline.py
```

File order:

- `src/training/training_pipeline.py`
- `config/config.yaml`
- `src/training/run_manager.py`
- `data/historical/final_processed/train_data.parquet`
- `data/historical/final_processed/val_data.parquet`
- `data/historical/final_processed/test_data.parquet`
- `models/runs/<run_id>/*.ckpt`
- `models/config/<run_id>.yaml`
- `models/ACTIVE_MODEL.txt` when active run is set

Workflow:

- Load config and split parquet files.
- Drop configured audit columns and missing-value marker columns.
- Validate required training columns.
- Build PyTorch Forecasting `TimeSeriesDataSet` objects.
- Create Temporal Fusion Transformer model.
- Train with Lightning callbacks.
- Save checkpoints.
- Test the trained model.
- Update run metadata.
- Optionally resume from an earlier run based on policy.

Important current dependency note:

- Training imports `lightning.pytorch`.
- `requirements.txt` lists `pytorch-lightning`, not `lightning`.
- A clean install may need dependency updates before this script runs successfully.

## 9. Testing and Evaluation Workflow

Primary test script:

```powershell
python src/testing/testing_pipeline.py
```

File order:

- `src/testing/testing_pipeline.py`
- `config/config.yaml`
- `src/training/training_pipeline.py`
- `data/historical/final_processed/test_data.parquet`
- `models/runs/<run_id>/*.ckpt`
- `models/testing/<run_id>_epoch1_test/test_predictions_vs_actual.csv`
- `models/testing/<run_id>_epoch1_test/metrics.json`

Summary/export scripts:

- `src/testing/summarize_results.py`
- `src/testing/export_forecast_summary.py`
- `src/evaluation/evaluation.py`
- `src/evaluation/results_exporter.py`

Workflow:

- Load trained model checkpoint.
- Rebuild compatible test dataset.
- Generate predictions.
- Compare predictions against actual load.
- Save metrics and prediction-vs-actual CSV.
- Optional exporters create smaller forecast evaluation bundles.

## 10. Scaffolded or Secondary Workflows

### Placeholder inference engine

Files:

- `src/forecast/forecast_engine.py`
- `src/forecast/main_inference.py`
- `src/pipelines/run_full_system.py`

These files provide a simpler scaffolded inference/evaluation path. The live backend forecast route does not use `ForecastEngine.generate_forecast()`. The live backend uses `src/forecast/tft_inference.py`.

### Full system runner

Command:

```powershell
python src/pipelines/run_full_system.py
```

Current behavior:

- Checks for prepared splits.
- Runs data prep if splits are missing.
- Checks for an active model or final checkpoint.
- Runs scaffolded inference/evaluation.
- Attempts to launch a Streamlit dashboard at `src/streamlit_frontend/streamlit_app.py`.

Important current code note:

- `src/streamlit_frontend/streamlit_app.py` is not present in this repository scan.
- The active frontend is the React/Vite app under `frontend/`.

## 11. Roles and How Personnel Connect

### Administrator

Current implemented work:

- Seeds or creates initial users through `src/auth/seed_admin.py`.
- Logs into frontend using admin credentials.
- Can create invite tokens through `POST /api/auth/admin/invite`.
- Can list users through `GET /api/auth/admin/users`.
- Can change user roles through `PUT /api/auth/admin/users/{user_id}/role`.
- Can deactivate users through `DELETE /api/auth/admin/users/{user_id}`.

Current limitation:

- There is no implemented frontend admin page route. Admin operations exist in backend API code but are not exposed through a working `/admin` page in `App.tsx`.

### Analyst

Current implemented work:

- Can log in.
- Can use the dashboard if authenticated.
- Backend has `require_analyst_or_admin()` middleware available for future protected analytics routes.

Current limitation:

- The live forecast endpoint is not currently protected by role middleware.
- No analyst-specific frontend route exists.

### Operator

Current implemented work:

- Newly registered users are assigned `OPERATOR`.
- Operators can log in and use the protected dashboard.
- Operators can select forecast dates and temperature scenarios.

### Model developer

Current implemented work:

- Runs historical preparation.
- Runs ingestion when operational snapshots are needed.
- Runs training.
- Runs testing and export scripts.
- Reviews model checkpoints and metrics.

### Frontend developer

Current implemented work:

- Works under `frontend/`.
- Maintains pages, auth client, dashboard chart rendering, sidebar, and visual components.

### Backend developer

Current implemented work:

- Maintains `backend/main.py`.
- Maintains auth routes and database logic.
- Maintains forecast API integration.

## 12. Important Current Gaps and Mismatches

These are not failures of the whole project, but they matter for onboarding.

- `requirements.txt` is missing several imports used by current code, including FastAPI, requests, requests-cache, BeautifulSoup, holidays, PyJWT, python-dotenv, passlib, pydantic email support, and lxml.
- `src/training/training_pipeline.py` imports `lightning.pytorch`, while `requirements.txt` lists `pytorch-lightning`.
- `frontend/src/pages/Dashboard.tsx` hardcodes the forecast API URL instead of using `VITE_API_BASE_URL`.
- `docs/AUTH_SYSTEM.md` documents admin endpoints as `/api/admin/...`, but current code uses `/api/auth/admin/...`.
- `docs/data/PREPROCESSING_HARDENING.md` describes validators that are not present in current `src/data_quality.py`.
- `frontend/src/components/UserMenu.tsx` links to `/admin`, but there is no `/admin` route in `App.tsx`.
- Some dashboard/demo copy mentions capabilities like peak alerts or protected analytics endpoints. The current implemented dashboard shows forecast visualization and metrics; it does not implement alerts.
- `src/pipelines/run_full_system.py` tries to launch a Streamlit frontend file that is not present.
- `src/forecast/tft_inference.py` may choose the lexicographically last final checkpoint rather than the checkpoint with the best validation loss.
- `.gitignore` ignores most model and data artifacts, but this repository scan includes data and model artifacts. Treat them as present local artifacts, but be careful when moving the repo to a fresh clone.

## 13. File-by-File Inventory

### Root files

- `.env.example`: Documents environment variables for JWT secrets, token lifetimes, and dummy fallback behavior.
- `.gitignore`: Excludes virtual environments, secrets, generated data, model artifacts, frontend build outputs, caches, and logs.
- `LICENSE`: Project license file.
- `README.md`: Main setup and usage guide for cloning, installing, seeding auth, and running backend/frontend.
- `requirements.txt`: Python dependency list. Currently incomplete relative to imports used in the codebase.
- `test_auth.py`: Small auth smoke-test script that initializes the database, checks admin login, and validates JWT creation/verification.

### Backend

- `backend/main.py`: FastAPI application entry point. Defines CORS, health route, forecast route, auth router registration, forecast scaling, metrics calculation, actuals attachment, and dummy fallback behavior.

### Configuration

- `config/config.yaml`: Central config for data paths, ingestion settings, model hyperparameters, feature windows, training settings, and checkpoint directories.

### Auth system

- `src/auth/__init__.py`: Package marker for auth module.
- `src/auth/database.py`: SQLite schema creation and database access layer for users, roles, invite tokens, and refresh tokens.
- `src/auth/jwt_handler.py`: Creates, verifies, and decodes JWT access and refresh tokens.
- `src/auth/middleware.py`: FastAPI dependencies for current user lookup and role enforcement.
- `src/auth/models.py`: Pydantic request and response models for auth and admin APIs.
- `src/auth/routes.py`: FastAPI auth and admin route definitions.
- `src/auth/seed_admin.py`: Creates/reset demo admin, analyst, and operator users.

### Ingestion

- `src/ingestion/__init__.py`: Exports `run_ingestion`.
- `src/ingestion/load_fetcher.py`: Scrapes Delhi SLDC load data, caches day-level files, normalizes timestamps/load, and resamples to 15-minute cadence.
- `src/ingestion/pipeline.py`: Runs operational ingestion by fetching load, weather, holidays, merging data, and writing snapshots/manifests.
- `src/ingestion/weather_fetcher.py`: Fetches Open-Meteo archive/forecast weather data and resamples it to 15-minute cadence.

### Data preparation and quality

- `src/feature_engineer.py`: Adds time, cyclical, lag, and rolling features to load/weather data.
- `src/dataset_builder.py`: Cleans raw/featured data, fills missing values, creates chronological splits, and builds PyTorch Forecasting datasets.
- `src/data_quality.py`: Generates stage-level quality summaries, null counts, row flow, and preprocessing summary output.

### Forecasting

- `src/forecast/fallback_data.py`: Generates fallback forecast JSON when enabled and real inference fails.
- `src/forecast/forecast_engine.py`: Placeholder/scaffold forecast engine, not used by the live backend route.
- `src/forecast/main_inference.py`: CLI scaffold for loading a model, running inference, evaluating, and exporting outputs.
- `src/forecast/peak_detection.py`: Finds peak forecast value and timestamp from prediction data.
- `src/forecast/tft_inference.py`: Operational TFT inference path used by the live backend forecast route.

### Training

- `src/training/run_manager.py`: Creates training run directories, stores run config/metadata, tracks active model pointer, and resolves checkpoints.
- `src/training/tft_model.py`: Custom LightningModule wrapper around Temporal Fusion Transformer. Present as model abstraction, but main training currently builds TFT directly in `training_pipeline.py`.
- `src/training/training_pipeline.py`: Main training script for loading splits, building datasets/dataloaders, training TFT, saving checkpoints, and testing.

### Testing

- `src/testing/__init__.py`: Package marker for testing module.
- `src/testing/export_forecast_summary.py`: Reads testing outputs and writes compact evaluation summaries.
- `src/testing/summarize_results.py`: Prints quick summary statistics for saved test prediction CSV.
- `src/testing/testing_pipeline.py`: Runs checkpoint evaluation against test data and saves metrics/prediction outputs.

### Evaluation

- `src/evaluation/evaluation.py`: Metric functions and `ModelEvaluator` for MAE, RMSE, MAPE, and SMAPE.
- `src/evaluation/results_exporter.py`: Saves prediction CSV, metrics JSON, and peak JSON.

### Pipelines

- `src/pipelines/main_ingestion.py`: CLI wrapper around operational ingestion.
- `src/pipelines/prepare_historical_data.py`: Main historical data preparation script.
- `src/pipelines/run_full_system.py`: Orchestrates prep, training check, scaffolded inference/evaluation, and an attempted Streamlit dashboard launch.

### Shared utilities

- `src/shared/artifact_repository.py`: Central path and artifact helper for saving/loading datasets and checkpoints.
- `src/shared/config.py`: Reads simple environment-level settings, currently `ENABLE_DUMMY_FALLBACK`.

### Frontend app shell and config

- `frontend/package.json`: Frontend dependencies and scripts.
- `frontend/package-lock.json`: Locked npm dependency tree.
- `frontend/index.html`: HTML entry point loaded by Vite.
- `frontend/vite.config.ts`: Vite React config, alias setup, and dev server port `3001`.
- `frontend/tsconfig.json`: TypeScript strict compiler config.
- `frontend/vite-env.d.ts`: Vite TypeScript type declarations.
- `frontend/tailwind.config.ts`: Tailwind theme colors and content scanning paths.
- `frontend/postcss.config.mjs`: PostCSS plugin config for Tailwind and Autoprefixer.
- `frontend/eslint.config.mjs`: ESLint flat config for JS/TS/TSX files.

### Frontend source

- `frontend/src/main.tsx`: React entry point that mounts `App`.
- `frontend/src/App.tsx`: Router, auth bootstrap, protected dashboard route, login/register routes.
- `frontend/src/index.css`: Global CSS, Tailwind layers, theme variables, layout and animation styles.
- `frontend/src/hooks/useAuth.ts`: Zustand auth store, token persistence, login/register/logout/refresh helpers.
- `frontend/src/lib/authClient.ts`: Axios client for auth APIs, bearer token injection, and refresh-on-401 behavior.
- `frontend/src/lib/utils.ts`: `cn()` helper that combines `clsx` with `tailwind-merge`.
- `frontend/src/pages/Login.tsx`: Login form and demo credential display.
- `frontend/src/pages/Register.tsx`: Invite-based registration form.
- `frontend/src/pages/Dashboard.tsx`: Main authenticated dashboard, forecast API call, scenario controls, chart rendering, metrics, and page sections.
- `frontend/src/components/ProtectedRoute.tsx`: Route guard for authentication and optional role checks.
- `frontend/src/components/UserMenu.tsx`: User dropdown with role badge, logout, and an admin link. Currently not used by the app route tree.

### Frontend visual components

- `frontend/src/components/aceternity/background-beams-demo.tsx`: Demo wrapper for animated beam background content.
- `frontend/src/components/aceternity/layout-text-flip-demo.tsx`: Demo wrapper for rotating text copy.
- `frontend/src/components/aceternity/macbook-scroll-demo.tsx`: Demo wrapper for MacBook scroll visual.
- `frontend/src/components/aceternity/sidebar-demo.tsx`: Defines dashboard sidebar items and wraps dashboard content in sidebar layout.
- `frontend/src/components/aceternity/timeline-demo.tsx`: Demo wrapper for project timeline content.
- `frontend/src/components/features/functional-footer.tsx`: Footer with navigation actions, API docs link, mail link, and status copy.
- `frontend/src/components/ui/background-beams.tsx`: Animated background beam component using Framer Motion.
- `frontend/src/components/ui/layout-text-flip.tsx`: Animated rotating text component.
- `frontend/src/components/ui/macbook-scroll.tsx`: Scroll-linked laptop mockup component.
- `frontend/src/components/ui/sidebar.tsx`: Responsive sidebar component with desktop and mobile behavior.
- `frontend/src/components/ui/timeline.tsx`: Vertical timeline rendering component.

### Frontend static assets

- `frontend/public/favicon.ico`: Browser favicon.
- `frontend/public/favicon.svg`: SVG favicon source.

### Documentation

- `docs/AUTH_SYSTEM.md`: Auth system documentation. Some endpoint paths are stale relative to current code.
- `docs/data/PREPROCESSING_HARDENING.md`: Preprocessing hardening documentation. Some described validator classes are not present in current code.
- `docs/PROJECT_WORKFLOW_REPORT.md`: This onboarding and workflow report.

### Historical data lane

- `data/historical/README.md`: Describes historical data as the locked model-development corpus.
- `data/historical/raw/README.md`: Describes raw historical source layout.
- `data/historical/raw/electricity_demand_2021-01-01_to_2026-04-06.csv`: Merged historical load/weather/holiday dataset used as preparation input.
- `data/historical/raw/ingestion_manifest_2021-01-01_to_2026-04-06.json`: Manifest and metadata for the historical raw snapshot.
- `data/historical/raw/calendar/README.md`: Describes holiday/calendar raw source files.
- `data/historical/raw/calendar/india_holidays_2021-01-01_to_2026-04-06.csv`: India holiday calendar used for holiday enrichment.
- `data/historical/raw/demand_load/README.md`: Describes load-only historical source files.
- `data/historical/raw/demand_load/load_sldc_2021-01-01_to_2026-04-06.csv`: Historical SLDC load series.
- `data/historical/raw/demand_load/powerdemand_5min_2021_to_2024_load_only.csv`: Base historical load-only demand file.
- `data/historical/raw/weather/README.md`: Describes weather-only historical source files.
- `data/historical/raw/weather/weather_openmeteo_2021-01-01_to_2026-04-06.csv`: Historical Open-Meteo weather source.
- `data/historical/feature_engineered/README.md`: Describes the feature-engineered lane.
- `data/historical/feature_engineered/featured_data.parquet`: Feature-engineered historical dataset.
- `data/historical/final_processed/README.md`: Describes final cleaned and split training data.
- `data/historical/final_processed/cleaned_data.parquet`: Cleaned and imputed historical data before splitting.
- `data/historical/final_processed/data_quality_report.json`: Stage-wise data quality report.
- `data/historical/final_processed/prep_metadata.json`: Preparation metadata and split details.
- `data/historical/final_processed/train_data.parquet`: Chronological training split.
- `data/historical/final_processed/val_data.parquet`: Chronological validation split.
- `data/historical/final_processed/test_data.parquet`: Chronological test split.

### Operational data lane

- `data/operational/README.md`: Describes operational data as ephemeral inference-only data.

Runtime-created operational files may include:

- `data/operational/recent_load.csv`
- `data/operational/raw/load_sldc_<date>.csv`
- `data/operational/raw/load_sldc_<start_date>_to_<end_date>.csv`
- `data/operational/raw/weather_openmeteo_<start_date>_to_<end_date>.csv`
- `data/operational/raw/electricity_demand_<start_date>_to_<end_date>.csv`
- `data/operational/raw/ingestion_manifest_<start_date>_to_<end_date>.json`

### Model configs and checkpoints

- `models/config/20260408_070201.yaml`: Saved training config for run `20260408_070201`.
- `models/config/20260408_075759.yaml`: Saved training config for run `20260408_075759`.
- `models/final model/checkpoint_summary.json`: Summary of final checkpoint hyperparameters.
- `models/final model/epoch=epoch=00-val_loss=val_loss=59.31.ckpt`: Final-model checkpoint artifact.
- `models/final model/epoch=epoch=01-val_loss=val_loss=96.36.ckpt`: Final-model checkpoint artifact currently likely selected by lexicographic order.
- `models/runs/20260408_070201/epoch=epoch=00-val_loss=val_loss=172.62.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=01-val_loss=val_loss=96.36.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=02-val_loss=val_loss=118.37.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=03-val_loss=val_loss=146.38.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=04-val_loss=val_loss=155.81.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=05-val_loss=val_loss=200.43.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=06-val_loss=val_loss=153.50.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=07-val_loss=val_loss=168.12.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/epoch=epoch=08-val_loss=val_loss=229.22.ckpt`: Training checkpoint for run `20260408_070201`.
- `models/runs/20260408_070201/last.ckpt`: Last checkpoint for run `20260408_070201`.
- `models/runs/20260408_075759/epoch=epoch=00-val_loss=val_loss=188.41.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=01-val_loss=val_loss=203.45.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=02-val_loss=val_loss=245.15.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=03-val_loss=val_loss=166.25.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=04-val_loss=val_loss=112.94.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=05-val_loss=val_loss=117.33.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/epoch=epoch=06-val_loss=val_loss=145.05.ckpt`: Training checkpoint for run `20260408_075759`.
- `models/runs/20260408_075759/last.ckpt`: Last checkpoint for run `20260408_075759`.
- `models/testing/20260408_070201_epoch1_test/metrics.json`: Saved metrics for a test evaluation run.
- `models/testing/20260408_070201_epoch1_test/test_predictions_vs_actual.csv`: Saved predictions and actuals for a test evaluation run.

## 14. Recommended Reading Order for a New Team Member

- Read `README.md` first to understand setup.
- Read `config/config.yaml` to understand paths, model settings, and workflow assumptions.
- Read `backend/main.py` to understand the live API surface.
- Read `frontend/src/App.tsx`, then `frontend/src/pages/Login.tsx`, `Register.tsx`, and `Dashboard.tsx`.
- Read `src/auth/routes.py`, `src/auth/database.py`, and `src/auth/jwt_handler.py` for authentication.
- Read `src/forecast/tft_inference.py` for live model inference.
- Read `src/ingestion/load_fetcher.py` and `src/ingestion/weather_fetcher.py` for real-time data acquisition.
- Read `src/pipelines/prepare_historical_data.py`, `src/dataset_builder.py`, and `src/feature_engineer.py` for training data preparation.
- Read `src/training/training_pipeline.py` and `src/testing/testing_pipeline.py` for model development.

## 15. Mental Model to Keep

The project has two lanes:

- Runtime lane: frontend -> backend -> live SLDC/weather data -> final TFT checkpoint -> dashboard response.
- Model-development lane: historical CSV -> feature engineering -> cleaned/split parquet -> training checkpoints -> testing outputs.

The runtime lane should stay operational and fast. The model-development lane should stay reproducible and carefully versioned.
