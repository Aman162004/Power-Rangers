"""Real TFT model inference for operational forecasting."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import torch
import yaml
import requests
import logging

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", module="pytorch_forecasting")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'loss' is an instance of `nn\.Module`",
    module=r"lightning\.pytorch\.utilities\.parsing",
)
warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'logging_metrics' is an instance of `nn\.Module`",
    module=r"lightning\.pytorch\.utilities\.parsing",
)

from src.ingestion.load_fetcher import fetch_sldc_load_data
from src.ingestion.weather_fetcher import fetch_openmeteo_weather_data
import holidays as holidays_lib


def _load_config(config_path: str | Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _register_safe_globals() -> None:
    """Register pytorch_forecasting classes for safe checkpoint loading."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    from pytorch_forecasting.data.encoders import (
        EncoderNormalizer,
        GroupNormalizer,
        MultiNormalizer,
        NaNLabelEncoder,
        TorchNormalizer,
    )

    np_core = getattr(np, "_core", None)
    if np_core is None:
        np_core = np.core
    
    add_safe_globals(
        [
            EncoderNormalizer,
            GroupNormalizer,
            MultiNormalizer,
            NaNLabelEncoder,
            TorchNormalizer,
            np_core.multiarray.scalar,
            np.dtype,
            type(np.dtype("float64")),
            type(np.dtype("float32")),
        ]
    )


def _find_latest_checkpoint(config: dict) -> Path:
    """Find the latest checkpoint in models/ directory."""
    models_root = Path(config.get("data", {}).get("models_root", "models"))
    if not models_root.is_absolute():
        models_root = PROJECT_ROOT / models_root
    
    # Check final model directory first
    final_model_dir = models_root / "final model"
    if final_model_dir.exists():
        candidates = sorted(final_model_dir.glob("*.ckpt"))
        if candidates:
            return candidates[-1]
    
    # Fall back to runs directory - pick checkpoint with lowest val loss
    runs_dir = models_root / "runs"
    if runs_dir.exists():
        candidates = sorted(runs_dir.rglob("epoch=*.ckpt"))
        if candidates:
            def sort_key(p: Path) -> float:
                try:
                    return float(p.name.split("val_loss=")[-1].replace(".ckpt", ""))
                except (ValueError, IndexError):
                    return float("inf")
            candidates.sort(key=sort_key)
            return candidates[0]
    
    raise FileNotFoundError("No checkpoint found")


def _load_tft_model(checkpoint: dict, prediction_dataset: TimeSeriesDataSet, config: dict) -> TemporalFusionTransformer:
    """Load TFT checkpoint using the shipped prediction dataset metadata.

    NOTE: This function is the SERVING path. Callers should have already
    invoked `validate_checkpoint_geometry(checkpoint_path, config)` before
    passing the checkpoint dict here — see run_tft_inference() below.
    The geometry guard is intentionally not re-run inside _load_tft_model
    because the checkpoint dict has already been loaded at this point and
    we only have the state_dict (no path).
    """
    model = TemporalFusionTransformer.from_dataset(
        prediction_dataset,
        hidden_size=config["model"]["hidden_size"],
        attention_head_size=config["model"]["attention_head_size"],
        dropout=config["model"]["dropout"],
        hidden_continuous_size=config["model"]["hidden_continuous_size"],
        output_size=len(config["model"]["quantiles"]),
        loss=QuantileLoss(quantiles=config["model"]["quantiles"]),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run_tft_inference(
    config_path: str | Path = "config/config.yaml",
    checkpoint_path: str | Path | None = None,
    historical_days: int = 10,
    forecast_date: str | None = None,
    load_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run TFT inference. Raises on failure for graceful backend fallback.
    
    Returns:
        DataFrame with 'timestamp', 'p10', 'p50', 'p90' columns
    """
    _register_safe_globals()
    config = _load_config(config_path)
    
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(config)
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # === Geometry guard (Step 5) ===
    # Refuse to silently resume into a checkpoint trained under a different
    # encoder/decoder window, lag list, or rolling-window list than the active
    # config. See src/training/run_manager.py::validate_checkpoint_geometry.
    from src.training.run_manager import validate_checkpoint_geometry
    validate_checkpoint_geometry(checkpoint_path, config)

    print(f"[TFT] Loading checkpoint from {checkpoint_path.name}")
    
    # Resolve forecast date
    if forecast_date:
        target_dt = datetime.strptime(forecast_date, "%Y-%m-%d")
        hist_end = datetime.combine(target_dt.date() - timedelta(days=1), datetime.min.time())
    else:
        hist_end = datetime.now()
    
    hist_start = hist_end - timedelta(days=historical_days)
    start_str = hist_start.strftime("%Y-%m-%d")
    end_str = hist_end.strftime("%Y-%m-%d")
    
    # Fetch data only when not provided by caller.
    if load_df is None:
        print(f"[TFT] Fetching operational data")
        load_df = fetch_sldc_load_data(start_str, end_str)

    if load_df.empty:
        raise RuntimeError("No load data available")
    
    cfg_ing = config.get("ingestion", {})
    forecast_end = hist_end + timedelta(minutes=15 * config.get("pipeline", {}).get("decoder_window", 192))
    forecast_end_str = forecast_end.strftime("%Y-%m-%d")

    weather_source = "openmeteo"
    try:
        weather_df = fetch_openmeteo_weather_data(
            start_str,
            forecast_end_str,
            latitude=float(cfg_ing.get("latitude", 28.6139)),
            longitude=float(cfg_ing.get("longitude", 77.2090)),
            timezone=cfg_ing.get("timezone", "Asia/Kolkata"),
        )
        if weather_df.empty:
            weather_source = "fallback"
            logger.warning(
                "[TFT] Weather fetch returned empty — falling back to constant weather features. "
                "Forecast will NOT reflect temperature/humidity/wind/rain drivers."
            )
    except requests.RequestException as exc:
        weather_source = "fallback"
        logger.warning(
            "[TFT] Weather fetch failed (%s) — continuing with fallback constant weather features. "
            "Forecast will NOT reflect temperature/humidity/wind/rain drivers.",
            exc,
        )
        weather_df = pd.DataFrame(columns=["timestamp", "temperature", "humidity", "wind_speed", "rainfall"])

    avg_temperature_c = None
    if not weather_df.empty and "temperature" in weather_df.columns:
        temperature_series = pd.to_numeric(weather_df["temperature"], errors="coerce").dropna()
        if not temperature_series.empty:
            avg_temperature_c = float(temperature_series.mean())
    
    # Prepare inference data
    df = load_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").dropna()
    
    # Merge weather
    if not weather_df.empty:
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
        df = df.merge(weather_df[["timestamp", "temperature", "humidity", "wind_speed", "rainfall"]], 
                      on="timestamp", how="left")
    
    for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
        if col not in df.columns:
            df[col] = 25.0 if col == "temperature" else (50.0 if col == "humidity" else (5.0 if col == "wind_speed" else 0.0))
        df[col] = df[col].ffill().bfill().fillna(df[col].mean())

    # Add temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    hour_rad = 2 * np.pi * df["hour"] / 24.0
    df["sin_hour"] = np.sin(hour_rad)
    df["cos_hour"] = np.cos(hour_rad)
    
    # Holiday
    min_year, max_year = df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max()
    ind_holidays = holidays_lib.India(years=range(min_year, max_year + 1))
    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: int(d in ind_holidays))
    
    # Lags and rolling means (driven by config.features.{lags,rolling_windows})
    df = df.sort_values("timestamp").reset_index(drop=True)
    lag_values = config.get("features", {}).get("lags", [4, 24, 96])
    rolling_values = config.get("features", {}).get("rolling_windows", [4, 24])
    for lag in lag_values:
        df[f"load_lag_{lag}"] = df["load_mw"].shift(lag)
    for win in rolling_values:
        df[f"rolling_mean_{win}"] = df["load_mw"].rolling(win, min_periods=1).mean()
    df = df.ffill().bfill()

    # Create inference frame compatible with the training dataset schema.
    df["time_idx"] = range(len(df))
    df["group_id"] = 0

    lag_cols = [f"load_lag_{lag}" for lag in lag_values]
    rolling_cols = [f"rolling_mean_{win}" for win in rolling_values]
    cols_required = [
        "time_idx", "group_id", "load_mw",
        "hour", "day_of_week", "month", "sin_hour", "cos_hour",
        "temperature", "humidity", "wind_speed", "rainfall", "is_holiday",
        *lag_cols,
        *rolling_cols,
    ]

    df_inf = df[["timestamp"] + cols_required].copy()

    # Get encoder / decoder window
    enc_win = config.get("pipeline", {}).get("encoder_window", 24)
    dec_win = config.get("pipeline", {}).get("decoder_window", 192)

    # History length must cover (encoder window + longest lag) so every row in
    # the encoder has a valid lag_X for the full horizon. Audit round 6 fix:
    # the old formula `max(enc_win + 96, enc_win)` truncated the lag_672 history
    # (the encoder saw 192 + 96 = 288 rows, but lag_672 needs 672 rows of
    # context). With encoder_window=192 + max(lag)=672 we need ≥ 864 rows of
    # history; `historical_days=10` provides 960 rows (96 rows of buffer).
    history_len = max(enc_win + (max(lag_values) if lag_values else 0), enc_win)
    df_hist = df_inf.iloc[-history_len:].copy()
    
    # Create future rows
    last_ts = df_inf["timestamp"].iloc[-1]
    future_rows = []
    india_holidays = holidays_lib.India(years=[last_ts.year, (last_ts + timedelta(days=dec_win * 15 // 60 // 24)).year])
    
    for i in range(1, dec_win + 1):
        ts = last_ts + timedelta(minutes=15 * i)
        weather_row = df.loc[df["timestamp"] == ts]
        if weather_row.empty:
            temperature = float(df_inf["temperature"].iloc[-1])
            humidity = float(df_inf["humidity"].iloc[-1])
            wind_speed = float(df_inf["wind_speed"].iloc[-1])
            rainfall = float(df_inf["rainfall"].iloc[-1])
        else:
            row0 = weather_row.iloc[0]
            temperature = float(row0.get("temperature", df_inf["temperature"].iloc[-1]))
            humidity = float(row0.get("humidity", df_inf["humidity"].iloc[-1]))
            wind_speed = float(row0.get("wind_speed", df_inf["wind_speed"].iloc[-1]))
            rainfall = float(row0.get("rainfall", df_inf["rainfall"].iloc[-1]))

        future_rows.append({
            "timestamp": ts,
            "time_idx": len(df_inf) - 1 + i,
            "group_id": 0,
            "load_mw": df_inf["load_mw"].iloc[-1],
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "sin_hour": np.sin(2 * np.pi * ts.hour / 24),
            "cos_hour": np.cos(2 * np.pi * ts.hour / 24),
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
            "is_holiday": int(ts.date() in india_holidays),
            # Lag proxies — average over the last `lag` rows of each lag column
            # (same pattern as the pre-audit code; the window scales with each lag).
            **{f"load_lag_{lag}":
               df_inf[f"load_lag_{lag}"].iloc[-lag:].mean()
               if len(df_inf) >= lag else df_inf["load_mw"].mean()
               for lag in lag_values},
            # Rolling-mean proxies — use the most recent value (lagged trailing
            # mean is constant across the decoder window in practice).
            **{f"rolling_mean_{win}": df_inf[f"rolling_mean_{win}"].iloc[-1]
               for win in rolling_values},
        })
    
    df_future = pd.DataFrame(future_rows)
    df_full = pd.concat([df_hist, df_future], ignore_index=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_parameters = checkpoint.get("dataset_parameters")
    if not isinstance(dataset_parameters, dict):
        raise RuntimeError("Checkpoint is missing dataset parameters required for inference")

    prediction_dataset = TimeSeriesDataSet.from_parameters(
        dataset_parameters,
        df_full[cols_required],
        predict=True,
        stop_randomization=True,
    )

    model = _load_tft_model(checkpoint, prediction_dataset, config)

    # Run inference using model.predict so outputs are transformed back to MW scale.
    print(f"[TFT] Running model inference ({dec_win} steps)")
    prediction_loader = prediction_dataset.to_dataloader(train=False, batch_size=1)

    quantile_pred = model.predict(prediction_loader, mode="quantiles")
    if not torch.is_tensor(quantile_pred):
        raise RuntimeError(f"Unexpected quantile prediction type: {type(quantile_pred)!r}")

    quantile_np = quantile_pred.detach().cpu().numpy()
    if quantile_np.ndim != 3 or quantile_np.shape[0] < 1 or quantile_np.shape[2] < 3:
        raise RuntimeError(f"Unexpected quantile prediction shape: {quantile_np.shape}")

    output_np = quantile_np[0]
    horizon = min(len(df_future), output_np.shape[0])
    if horizon <= 0:
        raise RuntimeError("TFT prediction horizon is empty")

    results = df_future.iloc[:horizon][["timestamp"]].copy()
    results["p10"] = output_np[:horizon, 0]
    results["p50"] = output_np[:horizon, 1]
    results["p90"] = output_np[:horizon, 2]
    
    # Clip to positive
    for col in ["p10", "p50", "p90"]:
        if col in results.columns:
            results[col] = results[col].clip(lower=0).astype(float)

    results = results[["timestamp", "p10", "p50", "p90"]]
    results.attrs["avg_temperature_c"] = avg_temperature_c
    results.attrs["weather_source"] = weather_source

    return results
