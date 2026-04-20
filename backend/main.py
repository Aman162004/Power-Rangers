import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import sys
from pathlib import Path

# Add project root to python path to import src modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.load_fetcher import (
    ACTUAL_CACHE_FRESHNESS_WINDOW,
    ACTUAL_CACHE_PREFIX,
    fetch_sldc_actual_load_data,
    fetch_sldc_load_data,
)
from src.forecast.tft_inference import run_tft_inference
from src.forecast.fallback_data import load_dummy_forecast_data
from src.shared.config import ENABLE_DUMMY_FALLBACK
from src.auth.routes import router as auth_router
from src.auth.seed_admin import seed_admin

# Suppress TFT inference warnings during operation
warnings.filterwarnings("ignore", module="pytorch_forecasting")

app = FastAPI(title="Power Rangers Backend API")
logger = logging.getLogger(__name__)

FRONTEND_DIST_DIR = project_root / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
OPERATIONAL_DIR = Path(os.getenv("OPERATIONAL_DIR", "/tmp/power-rangers/operational"))

# Register auth routes
app.include_router(auth_router)


@app.on_event("startup")
def _startup() -> None:
    OPERATIONAL_DIR.mkdir(parents=True, exist_ok=True)
    seed_admin()


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def serve_root():
    if FRONTEND_INDEX_FILE.exists():
        return FileResponse(FRONTEND_INDEX_FILE)
    return {"status": "ok"}


def _get_now_ist() -> datetime:
    return datetime.now(pytz.timezone("Asia/Kolkata")).replace(tzinfo=None)


@app.post("/api/forecast")
def fetch_and_predict(
    days_to_fetch: int = 7,
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
    if days_to_fetch < 1:
        raise HTTPException(status_code=400, detail="days_to_fetch must be >= 1")

    # Temperature slider drives forecast scaling: 1 C maps to 2% load scaling.
    if temperature_delta_c is not None:
        if temperature_delta_c < -5 or temperature_delta_c > 5:
            raise HTTPException(status_code=400, detail="temperature_delta_c must be between -5 and 5")
        scenario_aggressiveness_pct = float(temperature_delta_c) * 2.0
    else:
        if aggressiveness_pct < -10 or aggressiveness_pct > 10:
            raise HTTPException(status_code=400, detail="aggressiveness_pct must be between -10 and 10")
        scenario_aggressiveness_pct = float(aggressiveness_pct)

    try:
        now = _get_now_ist()
        # 1. Fetch data
        # If forecast_date is provided, interpret it as the first day of forecast horizon.
        # Use previous day as the historical endpoint so predictions are generated for selected day.
        if forecast_date:
            forecast_target_date = _resolve_anchor_date(forecast_date).date()
            history_end = datetime.combine(forecast_target_date - timedelta(days=1), datetime.min.time())
        else:
            history_end = now
            forecast_target_date = None

        start_date = history_end - timedelta(days=days_to_fetch)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = history_end.strftime("%Y-%m-%d")

        should_prefetch_actuals = forecast_target_date is None or forecast_target_date <= now.date()

        with ThreadPoolExecutor(max_workers=1) as executor:
            actuals_future = None
            if should_prefetch_actuals:
                actuals_future = executor.submit(_warm_today_actual_cache)

            load_df = fetch_sldc_load_data(start_str, end_str)

            if load_df.empty:
                raise HTTPException(status_code=404, detail="No data fetched from SLDC.")

            # 2. Store in operational folder
            operational_file = OPERATIONAL_DIR / "recent_load.csv"
            load_df.to_csv(operational_file, index=False)

            # 3. Generate TFT forecast with real model
            forecast_df = _build_forecast_tft(
                load_df,
                forecast_date=forecast_date if forecast_date else end_str,
            )
            avg_temperature_c = forecast_df.attrs.get("avg_temperature_c")

            forecast_df = _apply_aggressiveness(forecast_df, scenario_aggressiveness_pct)

            if actuals_future is not None:
                try:
                    actuals_future.result()
                except Exception:
                    logger.exception("Background SLDC actual cache warm-up failed")

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

        # 4. Compute metrics from forecast-vs-actual overlap on the returned horizon
        metrics = _compute_forecast_metrics(forecast_with_actuals)

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
            "avg_temperature_c": avg_temperature_c,
            "message": "Data fetched and forecasts generated successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        if ENABLE_DUMMY_FALLBACK:
            logger.warning(f"Forecast generation failed, returning dummy data: {e}")
            return load_dummy_forecast_data()
        logger.exception("Forecast request failed")
        raise HTTPException(status_code=500, detail="Forecast generation failed. Check server logs.") from e


@app.get("/{requested_path:path}")
def serve_spa(requested_path: str):
    if requested_path == "api" or requested_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    if requested_path in {"docs", "openapi.json", "redoc"}:
        raise HTTPException(status_code=404, detail="Not found")

    if not FRONTEND_INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend bundle not built")

    frontend_root = FRONTEND_DIST_DIR.resolve()
    candidate = (FRONTEND_DIST_DIR / requested_path).resolve()

    try:
        candidate.relative_to(frontend_root)
    except ValueError:
        return FileResponse(FRONTEND_INDEX_FILE)

    if candidate.is_file():
        return FileResponse(candidate)

    if Path(requested_path).suffix:
        raise HTTPException(status_code=404, detail="Not found")

    return FileResponse(FRONTEND_INDEX_FILE)


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
        return _get_now_ist()

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
    now = _get_now_ist()
    today = now.date()

    if horizon_start.date() > today:
        out["actual_load_mw"] = pd.NA
        return out

    actual_start = horizon_start.strftime("%Y-%m-%d")
    actual_end = min(horizon_end.date(), today).strftime("%Y-%m-%d")
    actual_df = fetch_sldc_actual_load_data(actual_start, actual_end)

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


def _warm_today_actual_cache() -> None:
    now = _get_now_ist()
    today_str = now.strftime("%Y-%m-%d")
    try:
        fetch_sldc_load_data(
            today_str,
            today_str,
            cache_prefix=ACTUAL_CACHE_PREFIX,
            current_day_freshness_window=ACTUAL_CACHE_FRESHNESS_WINDOW,
        )
    except Exception:
        logger.exception("Failed to warm today's SLDC actual cache")


def _build_forecast_tft(load_df: pd.DataFrame, forecast_date: str | None = None) -> pd.DataFrame:
    """Run TFT inference to generate probabilistic forecasts.
    
    Args:
        load_df: Historical load data  
        forecast_date: Date to forecast for (YYYY-MM-DD format)
    
    Returns:
        DataFrame with timestamp, p10, p50, p90 columns
    """
    config_path = project_root / "config" / "config.yaml"

    # Run TFT inference only (no baseline fallback)
    forecast_df = run_tft_inference(
        config_path=str(config_path),
        checkpoint_path=None,  # Auto-finds latest
        historical_days=7,
        forecast_date=forecast_date,
        load_df=load_df,
    )

    # Ensure required columns exist and are numeric
    forecast_df = forecast_df.copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], errors="coerce")
    for col in ["p10", "p50", "p90"]:
        forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")
        forecast_df[col] = forecast_df[col].clip(lower=0.0)  # No negative loads

    return forecast_df.dropna(subset=["timestamp", "p10", "p50", "p90"])


def _apply_aggressiveness(forecast_df: pd.DataFrame, aggressiveness_pct: float) -> pd.DataFrame:
    if forecast_df.empty:
        return forecast_df

    scaled = forecast_df.copy()
    multiplier = 1.0 + (aggressiveness_pct / 100.0)
    for col in ("p10", "p50", "p90"):
        scaled[col] = pd.to_numeric(scaled[col], errors="coerce").fillna(0.0) * multiplier
        scaled[col] = scaled[col].clip(lower=0.0)
    return scaled


def _compute_forecast_metrics(forecast_df: pd.DataFrame) -> dict:
    """Compute metrics from forecast p50 vs available actuals on the forecast horizon.

    This function is intentionally defensive: metric errors should never fail the API.
    """
    try:
        if forecast_df.empty:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        clean = forecast_df.copy()
        clean["p50"] = pd.to_numeric(clean.get("p50"), errors="coerce")
        clean["actual_load_mw"] = pd.to_numeric(clean.get("actual_load_mw"), errors="coerce")
        clean = clean.dropna(subset=["p50", "actual_load_mw"]).reset_index(drop=True)

        if clean.empty:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        errors = (clean["actual_load_mw"] - clean["p50"]).abs()
        mae = float(errors.mean())
        rmse = float(((clean["actual_load_mw"] - clean["p50"]) ** 2).mean() ** 0.5)

        denom = clean["actual_load_mw"].abs().replace(0, pd.NA)
        mape_series = (errors / denom).dropna() * 100.0
        mape = float(mape_series.mean()) if not mape_series.empty else 0.0

        if not pd.notna(mae):
            mae = 0.0
        if not pd.notna(rmse):
            rmse = 0.0
        if not pd.notna(mape):
            mape = 0.0

        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
    except Exception:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
