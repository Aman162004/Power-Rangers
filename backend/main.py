import os
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests

import sys
from pathlib import Path

# Add project root to python path to import src modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.load_fetcher import fetch_sldc_load_data

app = FastAPI(title="Power Rangers Backend API")

# Setup CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPERATIONAL_DIR = project_root / "data" / "operational"
DEFAULT_HORIZON_STEPS = 192   # 192 × 15 min = 48 hours
DEFAULT_SEASONALITY_STEPS = 96  # 96 × 15 min = 24 hours (one full day seasonal cycle)

# Ensure operational directory exists
os.makedirs(OPERATIONAL_DIR, exist_ok=True)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/weather")
def get_weather(date: str | None = None):
    try:
        if date:
            # Try forecast API first
            url_forecast = f"https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&start_date={date}&end_date={date}&daily=temperature_2m_mean"
            response = requests.get(url_forecast, timeout=10)
            
            if response.status_code >= 400:
                # Fallback to archive API for past dates
                url_archive = f"https://archive-api.open-meteo.com/v1/archive?latitude=28.6139&longitude=77.2090&start_date={date}&end_date={date}&daily=temperature_2m_mean"
                response = requests.get(url_archive, timeout=10)
                
            response.raise_for_status()
            data = response.json()
            daily_temps = data.get("daily", {}).get("temperature_2m_mean")
            if daily_temps and len(daily_temps) > 0 and daily_temps[0] is not None:
                current_temp = daily_temps[0]
            else:
                current_temp = 25.0
        else:
            url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current_weather=true"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            current_temp = data.get("current_weather", {}).get("temperature", 25.0)

        return {"temperature_c": current_temp, "location": "Delhi"}
    except Exception as e:
        return {"temperature_c": 25.0, "location": "Delhi (Fallback)", "error": str(e)}


@app.post("/api/forecast")
def fetch_and_predict(
    days_to_fetch: int = 10,
    forecast_date: str | None = None,
    aggressiveness_pct: float = 0.0,
    temperature_delta_c: float | None = None,
):
    """
    1. Fetches recent load data using load_fetcher.
    2. Stores the data in the operational folder.
    3. Generates deterministic seasonal forecasts.
    4. Computes metrics from historical holdout rows.
    5. Joins actual observed values for predicted timestamps when available.
    6. Applies optional temperature/aggressiveness scenario scaling.
    7. Returns both historical data and predictions.
    """
    try:
        if days_to_fetch < 1:
            raise HTTPException(status_code=400, detail="days_to_fetch must be >= 1")

        # Temperature drives forecast scaling: 1 C delta maps to 2% load scaling.
        # Frontend sends delta = actualTempC - 25, so range 10-60°C → delta -15 to +35.
        if temperature_delta_c is not None:
            if temperature_delta_c < -15 or temperature_delta_c > 35:
                raise HTTPException(status_code=400, detail="temperature_delta_c must be between -15 and 35 (absolute temp 10–60 °C)")
            scenario_aggressiveness_pct = float(temperature_delta_c) * 2.0
        else:
            if aggressiveness_pct < -30 or aggressiveness_pct > 70:
                raise HTTPException(status_code=400, detail="aggressiveness_pct must be between -30 and 70")
            scenario_aggressiveness_pct = float(aggressiveness_pct)

        # 1. Fetch data
        # If forecast_date is provided, interpret it as the first day of forecast horizon.
        # Use previous day as the historical endpoint so predictions are generated for selected day.
        if forecast_date:
            forecast_target_date = _resolve_anchor_date(forecast_date).date()
            history_end = datetime.combine(forecast_target_date - timedelta(days=1), datetime.min.time())
        else:
            history_end = datetime.now()
            forecast_target_date = None

        start_date = history_end - timedelta(days=days_to_fetch)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = history_end.strftime("%Y-%m-%d")

        load_df = fetch_sldc_load_data(start_str, end_str)

        if load_df.empty:
            raise HTTPException(status_code=404, detail="No data fetched from SLDC.")

        # 2. Store in operational folder
        operational_file = OPERATIONAL_DIR / "recent_load.csv"
        load_df.to_csv(operational_file, index=False)

        # 3. Forecast using seasonal profile + trend
        forecast_df = _build_forecast(load_df, horizon_steps=DEFAULT_HORIZON_STEPS)
        forecast_df = _apply_aggressiveness(forecast_df, scenario_aggressiveness_pct)
        forecast_with_actuals = _attach_actuals_for_horizon(forecast_df)
        predictions = [
            {
                "timestamp": row.timestamp.isoformat(),
                "predicted_load_mw": float(row.p50),
                "p10": float(row.p10),
                "p50": float(row.p50),
                "p90": float(row.p90),
                "actual_load_mw": None if pd.isna(row.actual_load_mw) else float(row.actual_load_mw),
            }
            for row in forecast_with_actuals.itertuples(index=False)
        ]

        # 4. Compute metrics from holdout performance
        metrics = _compute_metrics(load_df)

        # Format historical data for JSON response
        historical_data = (
            load_df.dropna().tail(96 * 3).to_dict(orient="records")
        )  # Return last 3 days
        for row in historical_data:
            row["timestamp"] = row["timestamp"].isoformat()

        # Generate response metrics and peak summary
        if predictions:
            peak_pred = max(predictions, key=lambda p: p["predicted_load_mw"])
            peak_val = round(peak_pred["predicted_load_mw"], 2)
            peak_time = peak_pred["timestamp"].replace("T", " ")[:16]
        else:
            peak_val = 0
            peak_time = ""

        return {
            "historical": historical_data,
            "predictions": predictions,
            "metrics": metrics,
            "peak": {"peak_value": peak_val, "peak_timestamp": peak_time},
            "forecast_date": forecast_target_date.strftime("%Y-%m-%d") if forecast_target_date else end_str,
            "history_end_date": end_str,
            "aggressiveness_pct": scenario_aggressiveness_pct,
            "temperature_delta_c": temperature_delta_c,
            "message": "Data fetched and forecasts generated successfully.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _infer_step_minutes(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 15
    deltas = df["timestamp"].diff().dropna().dt.total_seconds().div(60)
    if deltas.empty:
        return 15
    step = int(round(deltas.median()))
    return step if step > 0 else 15


def _resolve_anchor_date(forecast_date: str | None) -> datetime:
    if not forecast_date:
        return datetime.now()

    try:
        return datetime.strptime(forecast_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="forecast_date must be in YYYY-MM-DD format") from exc


def _attach_actuals_for_horizon(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        out = forecast_df.copy()
        out["actual_load_mw"] = pd.NA
        return out

    out = forecast_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if out.empty:
        out["actual_load_mw"] = pd.NA
        return out

    horizon_start = out["timestamp"].min()
    horizon_end = out["timestamp"].max()
    today = datetime.now().date()

    if horizon_start.date() > today:
        out["actual_load_mw"] = pd.NA
        return out

    actual_start = horizon_start.strftime("%Y-%m-%d")
    actual_end = min(horizon_end.date(), today).strftime("%Y-%m-%d")
    actual_df = fetch_sldc_load_data(actual_start, actual_end)

    if actual_df.empty:
        out["actual_load_mw"] = pd.NA
        return out

    actual_df = actual_df.copy()
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"], errors="coerce")
    actual_df["actual_load_mw"] = pd.to_numeric(actual_df["load_mw"], errors="coerce")
    actual_df = actual_df.dropna(subset=["timestamp", "actual_load_mw"])
    actual_df = actual_df[
        (actual_df["timestamp"] >= horizon_start) & (actual_df["timestamp"] <= horizon_end)
    ]

    merged = out.merge(
        actual_df[["timestamp", "actual_load_mw"]].drop_duplicates(subset=["timestamp"]),
        on="timestamp",
        how="left",
    )
    return merged


def _build_forecast(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["load_mw"] = pd.to_numeric(clean["load_mw"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "load_mw"]).sort_values("timestamp")

    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "p10", "p50", "p90"])

    history = clean["load_mw"].reset_index(drop=True)
    last_ts = clean["timestamp"].iloc[-1]
    step_minutes = _infer_step_minutes(clean)

    seasonality = DEFAULT_SEASONALITY_STEPS
    has_seasonality = len(history) >= seasonality

    if len(history) >= seasonality * 2:
        trend_per_step = float((history.iloc[-seasonality:].mean() - history.iloc[-2 * seasonality : -seasonality].mean()) / seasonality)
    elif len(history) >= 2:
        trend_per_step = float((history.iloc[-1] - history.iloc[0]) / max(len(history) - 1, 1))
    else:
        trend_per_step = 0.0

    residuals = pd.Series(dtype=float)
    if has_seasonality:
        residuals = (history - history.shift(seasonality)).dropna()

    if len(residuals) >= 20:
        q10 = float(residuals.quantile(0.10))
        q90 = float(residuals.quantile(0.90))
    else:
        avg = float(history.tail(seasonality).mean()) if has_seasonality else float(history.iloc[-1])
        band = max(avg * 0.03, 50.0)
        q10 = -band
        q90 = band

    rows = []
    for i in range(1, horizon_steps + 1):
        if has_seasonality:
            base_idx = -seasonality + ((i - 1) % seasonality)
            seasonal_base = float(history.iloc[base_idx])
        else:
            seasonal_base = float(history.iloc[-1])

        p50 = max(seasonal_base + trend_per_step * i, 0.0)
        p10 = max(p50 + q10, 0.0)
        p90 = max(p50 + q90, p10)
        rows.append(
            {
                "timestamp": last_ts + timedelta(minutes=step_minutes * i),
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )

    return pd.DataFrame(rows)


def _apply_aggressiveness(forecast_df: pd.DataFrame, aggressiveness_pct: float) -> pd.DataFrame:
    if forecast_df.empty:
        return forecast_df

    scaled = forecast_df.copy()
    multiplier = 1.0 + (aggressiveness_pct / 100.0)
    for col in ("p10", "p50", "p90"):
        scaled[col] = pd.to_numeric(scaled[col], errors="coerce").fillna(0.0) * multiplier
        scaled[col] = scaled[col].clip(lower=0.0)
    return scaled


def _compute_metrics(df: pd.DataFrame) -> dict:
    clean = df.copy()
    clean["load_mw"] = pd.to_numeric(clean["load_mw"], errors="coerce")
    clean = clean.dropna(subset=["load_mw"]).reset_index(drop=True)

    if clean.empty:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    seasonality = DEFAULT_SEASONALITY_STEPS
    if len(clean) > seasonality:
        y_true = clean["load_mw"].iloc[seasonality:]
        y_pred = clean["load_mw"].shift(seasonality).iloc[seasonality:]
    elif len(clean) > 1:
        y_true = clean["load_mw"].iloc[1:]
        y_pred = clean["load_mw"].shift(1).iloc[1:]
    else:
        only = float(clean["load_mw"].iloc[0])
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0 if only == 0 else 0.0}

    errors = (y_true - y_pred).abs()
    mae = float(errors.mean())
    rmse = float(((y_true - y_pred) ** 2).mean() ** 0.5)
    denom = y_true.abs().replace(0, pd.NA)
    mape = float(((errors / denom).dropna() * 100.0).mean()) if not denom.dropna().empty else 0.0

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
