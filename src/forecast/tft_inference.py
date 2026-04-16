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

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

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
from src.training.training_pipeline import (
    _build_tft_datasets,
    _drop_unused_training_columns,
    _load_training_splits,
)
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
    """Resolve best checkpoint: prefer models/final model/, then best_model_path, then runs/ by val_loss."""
    models_root = Path(config.get("data", {}).get("models_root", "models"))
    if not models_root.is_absolute():
        models_root = PROJECT_ROOT / models_root

    # 1. Highest priority: any checkpoint placed in models/final model/
    final_model_dir = models_root / "final model"
    if final_model_dir.exists():
        candidates = sorted(final_model_dir.glob("*.ckpt"))
        if candidates:
            p = candidates[-1]
            print(f"[TFT] Auto-selected checkpoint from final model folder: {p.name}")
            return p

    # 2. Honour explicit best_model_path from config.yaml
    best_model_path = config.get("data", {}).get("best_model_path")
    if best_model_path:
        p = Path(best_model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p
        print(f"[TFT] WARNING: best_model_path '{p}' not found, falling back to runs/ scan.")

    models_root = Path(config.get("data", {}).get("models_root", "models"))
    if not models_root.is_absolute():
        models_root = PROJECT_ROOT / models_root

    # 2. Scan runs/ directory — pick checkpoint with lowest val loss
    #    Filenames may be 'epoch=epoch=10-val_loss=val_loss=138.47.ckpt'
    #    so we take the very last token after splitting on 'val_loss='
    runs_dir = models_root / "runs"
    if runs_dir.exists():
        candidates = [p for p in runs_dir.rglob("epoch=*.ckpt") if "last" not in p.name]
        if candidates:
            def sort_key(p: Path) -> float:
                try:
                    return float(p.name.split("val_loss=")[-1].replace(".ckpt", ""))
                except (ValueError, IndexError):
                    return float("inf")
            candidates.sort(key=sort_key)
            return candidates[0]

    raise FileNotFoundError("No checkpoint found — set data.best_model_path in config.yaml")


def _ensure_lag_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute any lag/rolling columns that are missing from the prepared parquet splits.

    This handles the case where the data pipeline was run with a smaller set of lags
    but the model was trained with additional ones (e.g. lag_672).  We recompute from
    the target column rather than failing with a KeyError inside TimeSeriesDataSet.
    """
    df = df.copy().sort_values("time_idx").reset_index(drop=True)
    cfg_lags = config.get("features", {}).get("lags", [4, 24, 96])
    cfg_rolling = config.get("features", {}).get("rolling_windows", [4, 24])

    for lag in cfg_lags:
        col = f"load_lag_{lag}"
        if col not in df.columns:
            df[col] = df["load_mw"].shift(lag).ffill().bfill()

    for w in cfg_rolling:
        col = f"rolling_mean_{w}"
        if col not in df.columns:
            df[col] = df["load_mw"].rolling(w, min_periods=1).mean()

    return df


def _load_tft_model(checkpoint_path: Path, config: dict) -> TemporalFusionTransformer:
    """Load TFT checkpoint and training dataset for consistent inference scaling."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild the training dataset exactly from prepared splits so encoders/scalers match training.
    train_df, val_df, test_df = _load_training_splits(config)
    train_df = _drop_unused_training_columns(train_df, config)
    val_df = _drop_unused_training_columns(val_df, config)
    test_df = _drop_unused_training_columns(test_df, config)

    # Patch any lag/rolling columns that were added to config after the data pipeline ran
    train_df = _ensure_lag_columns(train_df, config)
    val_df   = _ensure_lag_columns(val_df,   config)
    test_df  = _ensure_lag_columns(test_df,  config)

    train_dataset, _, _ = _build_tft_datasets(config, train_df, val_df, test_df)
    
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=config["model"]["hidden_size"],
        attention_head_size=config["model"]["attention_head_size"],
        dropout=config["model"]["dropout"],
        hidden_continuous_size=config["model"]["hidden_continuous_size"],
        output_size=len(config["model"]["quantiles"]),
        loss=QuantileLoss(quantiles=config["model"]["quantiles"]),
    )
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, train_dataset


def run_tft_inference(
    config_path: str | Path = "config/config.yaml",
    checkpoint_path: str | Path | None = None,
    historical_days: int = 7,
    forecast_date: str | None = None,
    load_df: pd.DataFrame | None = None,
    temperature_delta_c: float = 0.0,
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
    
    print(f"[TFT] Loading checkpoint from {checkpoint_path.name}")
    model, train_dataset = _load_tft_model(checkpoint_path, config)
    
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

    try:
        weather_df = fetch_openmeteo_weather_data(
            start_str,
            forecast_end_str,
            latitude=float(cfg_ing.get("latitude", 28.6139)),
            longitude=float(cfg_ing.get("longitude", 77.2090)),
            timezone=cfg_ing.get("timezone", "Asia/Kolkata"),
        )
    except requests.RequestException as exc:
        print(f"[TFT] Weather fetch failed, continuing with fallback features: {exc}")
        weather_df = pd.DataFrame(columns=["timestamp", "temperature", "humidity", "wind_speed", "rainfall"])

    avg_temperature_c = None
    if not weather_df.empty and "temperature" in weather_df.columns:
        temp_df = weather_df.copy()
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], errors="coerce")
        target_eval_date = target_dt.date() if forecast_date else datetime.now().date()
        target_day_weather = temp_df[temp_df["timestamp"].dt.date == target_eval_date]
        if not target_day_weather.empty:
            avg_temperature_c = float(pd.to_numeric(target_day_weather["temperature"], errors="coerce").mean())
        else:
            avg_temperature_c = float(pd.to_numeric(temp_df["temperature"], errors="coerce").mean())
            
        # Add delta so the frontend displays the exact user-adjusted temperature
        if avg_temperature_c is not None:
            avg_temperature_c += temperature_delta_c
    
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
    
    # Lags and rolling means — driven by config so they match training exactly
    cfg_lags = config.get("features", {}).get("lags", [4, 24, 96])
    cfg_rolling = config.get("features", {}).get("rolling_windows", [4, 24])

    df = df.sort_values("timestamp").reset_index(drop=True)
    for lag in cfg_lags:
        df[f"load_lag_{lag}"] = df["load_mw"].shift(lag)
    for w in cfg_rolling:
        df[f"rolling_mean_{w}"] = df["load_mw"].rolling(w, min_periods=1).mean()
    df = df.ffill().bfill()
    
    # Create inference frame compatible with the training dataset schema.
    df["time_idx"] = range(len(df))
    df["group_id"] = 0
    
    lag_cols = [f"load_lag_{lag}" for lag in cfg_lags]
    rolling_cols = [f"rolling_mean_{w}" for w in cfg_rolling]

    cols_required = [
        "time_idx", "group_id", "load_mw",
        "hour", "day_of_week", "month", "sin_hour", "cos_hour",
        "temperature", "humidity", "wind_speed", "rainfall", "is_holiday",
    ] + lag_cols + rolling_cols
    
    df_inf = df[["timestamp"] + cols_required].copy()
    
    # Get encoder window
    enc_win = config.get("pipeline", {}).get("encoder_window", 24)
    dec_win = config.get("pipeline", {}).get("decoder_window", 192)

    # Need at minimum max_lag + enc_win rows so no NaN remains in any lag column
    max_lag = max(cfg_lags) if cfg_lags else 96
    history_len = max(max_lag + enc_win, enc_win + 96)
    df_hist = df_inf.iloc[-history_len:].copy()
    # Re-index time_idx contiguously so pytorch_forecasting doesn't see gaps
    df_hist["time_idx"] = range(len(df_hist))
    
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

        # Apply what-if temperature adjustment
        temperature += temperature_delta_c

        # Fill lag/rolling values for future rows from recent history
        future_lag_vals = {}
        for lag in cfg_lags:
            col = f"load_lag_{lag}"
            future_lag_vals[col] = (
                df_inf[col].iloc[-lag:].mean() if len(df_inf) >= lag
                else df_inf["load_mw"].mean()
            )
        for w in cfg_rolling:
            col = f"rolling_mean_{w}"
            future_lag_vals[col] = float(df_inf[col].iloc[-1])

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
            **future_lag_vals,
        })
    
    df_future = pd.DataFrame(future_rows)
    df_full = pd.concat([df_hist, df_future], ignore_index=True)
    # Guarantee perfectly contiguous time_idx — pytorch_forecasting requires step=1
    df_full["time_idx"] = range(len(df_full))

    # Run inference using model.predict so outputs are transformed back to MW scale.
    print(f"[TFT] Running model inference ({dec_win} steps)")
    prediction_dataset = type(train_dataset).from_dataset(
        train_dataset,
        df_full[cols_required],
        predict=True,
        stop_randomization=True,
        allow_missing_timesteps=True,
    )
    prediction_loader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

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

    return results
