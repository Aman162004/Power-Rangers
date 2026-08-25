"""Evaluate a saved TFT checkpoint on the held-out test split and save comparison artifacts."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
import holidays as holidays_lib
from datetime import timedelta
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    module=r"sklearn\.utils\.validation",
)

from src.training.training_pipeline import (  # noqa: E402
    _build_tft_datasets,
    _drop_unused_training_columns,
    _load_training_splits,
    _register_safe_globals_for_checkpoint_resume,
    load_config,
)
from src.evaluation.evaluation import evaluate_all  # noqa: E402
from src.training.run_manager import TrainingRunManager  # noqa: E402


def _safe_load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a local trusted Lightning checkpoint with full objects allowed."""
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _select_checkpoint(config: dict[str, Any], run_id: str, checkpoint_name: str | None = None) -> Path:
    """Pick the checkpoint to evaluate from a run folder."""
    checkpoints_dir = Path(config['data']['models_root']) / config['data']['models_checkpoints_dir'].replace('models/', '').rstrip('/')
    run_dir = checkpoints_dir / run_id
    if checkpoint_name is not None:
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidate_files = sorted(run_dir.glob("epoch=*.ckpt"))
    if not candidate_files:
        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    # Prefer the checkpoint with the lowest validation loss in the filename.
    def sort_key(path: Path) -> float:
        name = path.name
        if "val_loss=" in name:
            try:
                return float(name.split("val_loss=")[-1].replace(".ckpt", ""))
            except ValueError:
                return float("inf")
        return float("inf")

    candidate_files = sorted(candidate_files, key=sort_key)
    return candidate_files[0]


def _build_model(config: dict[str, Any], train_dataset) -> TemporalFusionTransformer:
    """Rebuild the TFT architecture and load the saved state dict."""
    precision_mode = str(config.get('training', {}).get('precision', '32-true'))
    mask_bias = -1.0e4 if '16' in precision_mode else -1.0e9

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=config['model']['hidden_size'],
        attention_head_size=config['model']['attention_head_size'],
        dropout=config['model']['dropout'],
        hidden_continuous_size=config['model']['hidden_continuous_size'],
        output_size=len(config['model']['quantiles']),
        loss=QuantileLoss(quantiles=config['model']['quantiles']),
        mask_bias=mask_bias,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return model


def _metrics_from_predictions(actual: np.ndarray, p50: np.ndarray) -> dict[str, float]:
    """Compute the standard regression metrics against the median forecast."""
    summary = evaluate_all(actual, p50)
    return {
        "MAE": summary["mae"],
        "RMSE": summary["rmse"],
        "MAPE": summary["mape"],
        "SMAPE": summary["smape"],
    }


def _load_full_forecast_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Load the full prepared (train+val+test) frame with feature columns intact.

    The walk-forward backtest needs a single chronological frame with all
    feature columns so each origin can reuse the actual recorded values for its
    encoder context exactly as training saw them.
    """
    train_df, val_df, test_df = _load_training_splits(config)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = _drop_unused_training_columns(full_df, config)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
    full_df = full_df.dropna(subset=["timestamp"])
    full_df = full_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    full_df = full_df.reset_index(drop=True)
    full_df["time_idx"] = np.arange(len(full_df))
    full_df["group_id"] = 0
    return full_df


def _future_row_proxies(
    df_inf: pd.DataFrame,
    lag_values: list[int],
    rolling_values: list[int],
) -> dict[str, float]:
    """Build the serving-faithful lag/rolling proxy values for a future row.

    Mirrors `src/forecast/tft_inference.py::run_tft_inference`: each
    `load_lag_X` proxy is the mean of the last `X` values of that lag column,
    and each `rolling_mean_W` proxy is the most recent value. Using the same
    proxy here keeps the walk-forward backtest honest about how the deployed
    serving path actually constructs the decoder context.
    """
    proxies: dict[str, float] = {}
    for lag in lag_values:
        col = f"load_lag_{lag}"
        if col in df_inf.columns and len(df_inf) >= lag:
            proxies[col] = float(df_inf[col].iloc[-lag:].mean())
        else:
            proxies[col] = float(df_inf["load_mw"].mean())
    for win in rolling_values:
        col = f"rolling_mean_{win}"
        proxies[col] = float(df_inf[col].iloc[-1]) if col in df_inf.columns else 0.0
    return proxies


def _build_backtest_window(
    df_full: pd.DataFrame,
    origin_idx: int,
    history_len: int,
    horizon: int,
    lag_values: list[int],
    rolling_values: list[int],
) -> pd.DataFrame:
    """Construct a single predict-ready window ending at `origin_idx + horizon`.

    `df_full` must contain every feature column the model expects. The encoder
    portion is drawn from recorded history; the decoder portion is built with
    the same future-row proxies as the serving path.
    """
    past = df_full.iloc[origin_idx - history_len: origin_idx].copy()
    past = past.reset_index(drop=True)
    last_ts = past["timestamp"].iloc[-1]
    last_load = float(past["load_mw"].iloc[-1])

    proxies = _future_row_proxies(past, lag_values, rolling_values)
    india_holidays = holidays_lib.India(
        years=[last_ts.year, (last_ts + timedelta(minutes=15 * horizon)).year]
    )

    rows = []
    for i in range(1, horizon + 1):
        ts = last_ts + timedelta(minutes=15 * i)
        rows.append({
            "timestamp": ts,
            "load_mw": last_load,
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "sin_hour": np.sin(2 * np.pi * ts.hour / 24),
            "cos_hour": np.cos(2 * np.pi * ts.hour / 24),
            "is_holiday": int(ts.date() in india_holidays),
            "temperature": float(past["temperature"].iloc[-1]),
            "humidity": float(past["humidity"].iloc[-1]),
            "wind_speed": float(past["wind_speed"].iloc[-1]),
            "rainfall": float(past["rainfall"].iloc[-1]),
            **proxies,
        })
    future = pd.DataFrame(rows)

    window = pd.concat([past, future], ignore_index=True)
    window["time_idx"] = np.arange(len(window))
    window["group_id"] = 0
    return window


def walk_forward_backtest(
    config_path: str = "config/config.yaml",
    checkpoint_path: str | Path | None = None,
    stride: int = 96,
    horizon: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Run a full walk-forward backtest over the entire test window.

    Origins advance by `stride` (1 day) from `data.test_start_date` while a
    full `horizon` of recorded actuals remains. A 365-day test window with
    stride=96, horizon=192 yields 364 origins (one 48h forecast per day).

    Predictions mirror the serving construction in
    `tft_inference.py::run_tft_inference` exactly (encoder = recorded history,
    decoder = proxy rows). Returns a long-form DataFrame: origin, timestamp,
    p10, p50, p90, actual_load_mw, error_p50, abs_error_p50, ape_p50.
    """
    config = load_config(config_path)
    _register_safe_globals_for_checkpoint_resume()

    pipeline_cfg = config.get("pipeline", {})
    enc_win = int(pipeline_cfg.get("encoder_window", 192))
    horizon = int(horizon if horizon is not None else pipeline_cfg.get("decoder_window", 192))
    lag_values = list(config.get("features", {}).get("lags", []))
    rolling_values = list(config.get("features", {}).get("rolling_windows", []))
    history_len = max(enc_win + (max(lag_values) if lag_values else 0), enc_win)

    data_cfg = config.get("data", {})
    test_start = pd.Timestamp(data_cfg.get("test_start_date", "2025-04-07"))
    test_end = pd.Timestamp(data_cfg.get("test_end_date", "2026-04-06")) + pd.Timedelta(hours=23, minutes=45)

    from src.training.run_manager import validate_checkpoint_geometry
    if checkpoint_path is None:
        from src.forecast.tft_inference import _find_latest_checkpoint
        checkpoint_path = _find_latest_checkpoint(config)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"walk_forward_backtest: checkpoint not found at {checkpoint_path}")
    validate_checkpoint_geometry(checkpoint_path, config)

    print(f"[BACKTEST] Loading checkpoint {checkpoint_path.name}")
    checkpoint = _safe_load_checkpoint(checkpoint_path)

    full_df = _load_full_forecast_frame(config)
    print(f"[BACKTEST] Loaded {len(full_df)} rows of prepared data "
          f"({full_df['timestamp'].min()} -> {full_df['timestamp'].max()})")

    test_mask = (full_df["timestamp"] >= test_start) & (full_df["timestamp"] <= test_end)
    test_idx = np.where(test_mask.to_numpy())[0]
    if len(test_idx) == 0:
        raise ValueError(
            f"walk_forward_backtest: no rows in [{test_start}, {test_end}]. "
            "Check data.test_start_date / test_end_date against the prepared splits."
        )
    origin_idx = list(range(int(test_idx[0]), int(test_idx[-1]) - horizon + 1, stride))
    if not origin_idx:
        raise ValueError("walk_forward_backtest: no viable origins (test window shorter than horizon).")
    origins = [full_df["timestamp"].iloc[i] for i in origin_idx]
    print(f"[BACKTEST] {len(origin_idx)} origins across test window "
          f"[{origins[0]} -> {origins[-1]}], stride={stride}, horizon={horizon}")

    # Build a training dataset directly (mirrors _build_tft_datasets but only
    # needs the train split — avoids requiring a non-empty val/test split).
    from src.training.training_pipeline import _validate_required_columns
    train_df, _, _ = _load_training_splits(config)
    train_df = _drop_unused_training_columns(train_df, config)
    _validate_required_columns(train_df, 'train')
    lag_cols = [f'load_lag_{lag}' for lag in lag_values]
    rolling_cols = [f'rolling_mean_{win}' for win in rolling_values]
    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx='time_idx',
        target='load_mw',
        group_ids=['group_id'],
        min_encoder_length=max(enc_win // 2, 1),
        max_encoder_length=enc_win,
        min_prediction_length=1,
        max_prediction_length=horizon,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['hour', 'day_of_week', 'month', 'sin_hour', 'cos_hour',
                                  'temperature', 'humidity', 'wind_speed', 'rainfall', 'is_holiday'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['load_mw'] + lag_cols + rolling_cols,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    model = _build_model(config, train_dataset)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    dataset_parameters = checkpoint.get("dataset_parameters")
    if not isinstance(dataset_parameters, dict):
        raise RuntimeError("Checkpoint is missing dataset_parameters required for prediction")
    feature_cols = [c for c in full_df.columns]

    rows: list[dict[str, Any]] = []
    for n, (o_idx, origin_ts) in enumerate(zip(origin_idx, origins)):
        window = _build_backtest_window(
            full_df, o_idx, history_len, horizon, lag_values, rolling_values
        )
        prediction_dataset = TimeSeriesDataSet.from_parameters(
            dataset_parameters,
            window[feature_cols],
            predict=True,
            stop_randomization=True,
        )
        loader = prediction_dataset.to_dataloader(train=False, batch_size=1)

        with torch.no_grad():
            quantile_pred = model.predict(
                loader,
                mode="quantiles",
                mode_kwargs={"quantiles": config["model"]["quantiles"]},
                return_x=False,
                return_y=False,
                batch_size=1,
                trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "logger": False,
                    "enable_progress_bar": False,
                    "precision": "32-true",
                },
            )
        pred_np = quantile_pred.detach().cpu().numpy() if torch.is_tensor(quantile_pred) else np.asarray(quantile_pred)
        if pred_np.ndim == 3:
            pred_np = np.squeeze(pred_np, axis=0)
        if pred_np.ndim != 2 or pred_np.shape[0] < horizon:
            raise RuntimeError(
                f"[BACKTEST] Unexpected prediction shape {pred_np.shape} at origin {origin_ts}"
            )
        p10 = pred_np[:horizon, 0]
        p50 = pred_np[:horizon, 1] if pred_np.shape[1] > 1 else p10
        p90 = pred_np[:horizon, 2] if pred_np.shape[1] > 2 else p50

        actual_slice = full_df.iloc[o_idx: o_idx + horizon][["timestamp", "load_mw"]]
        if len(actual_slice) != horizon:
            break
        actual_vals = actual_slice["load_mw"].to_numpy(dtype=float)

        for step_i in range(horizon):
            act = float(actual_vals[step_i])
            pred = float(p50[step_i])
            rows.append({
                "origin": origin_ts,
                "timestamp": actual_slice["timestamp"].iloc[step_i],
                "p10": float(p10[step_i]),
                "p50": pred,
                "p90": float(p90[step_i]),
                "actual_load_mw": act,
                "error_p50": act - pred,
                "abs_error_p50": abs(act - pred),
                "ape_p50": abs(act - pred) / act * 100.0 if act != 0 else np.nan,
            })
        if (n + 1) % progress_every == 0 or n + 1 == len(origin_idx):
            print(f"[BACKTEST] {n + 1}/{len(origin_idx)} origins complete")

    backtest_df = pd.DataFrame(rows)
    print(f"[BACKTEST] Done — {len(backtest_df)} (origin x step) rows collected.")
    return backtest_df


def run_test_pipeline(
    config_path: str = "config/config.yaml",
    run_id: str = "20260408_070201",
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate the chosen checkpoint on the held-out test horizon and persist outputs."""
    config = load_config(config_path)
    _register_safe_globals_for_checkpoint_resume()

    print(f"[TEST] Using run: {run_id}")
    checkpoint_path = _select_checkpoint(config, run_id, checkpoint_name)
    print(f"[TEST] Using checkpoint: {checkpoint_path}")

    train_df, val_df, test_df = _load_training_splits(config)
    train_df = _drop_unused_training_columns(train_df, config)
    val_df = _drop_unused_training_columns(val_df, config)
    test_df = _drop_unused_training_columns(test_df, config)

    train_dataset, val_dataset, test_dataset = _build_tft_datasets(config, train_df, val_df, test_df)

    checkpoint = _safe_load_checkpoint(checkpoint_path)
    model = _build_model(config, train_dataset)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    requested_workers = config.get('training', {}).get('num_workers')
    if requested_workers is None:
        available_cpus = max(1, (os.cpu_count() or 4) - 1)
        num_workers = min(8, available_cpus)
    else:
        num_workers = max(0, int(requested_workers))

    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': bool(config.get('training', {}).get('pin_memory', True)),
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = bool(config.get('training', {}).get('persistent_workers', True))
        dataloader_kwargs['prefetch_factor'] = int(config.get('training', {}).get('prefetch_factor', 4))

    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=config['model']['batch_size'],
        **dataloader_kwargs,
    )

    raw_predictions = model.predict(
        test_dataloader,
        mode="quantiles",
        mode_kwargs={"quantiles": config['model']['quantiles']},
        return_x=False,
        return_y=False,
        batch_size=config['model']['batch_size'],
        trainer_kwargs={
            "accelerator": 'gpu' if torch.cuda.is_available() else 'cpu',
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "precision": '32-true',
        },
    )

    predictions = raw_predictions.detach().cpu().numpy() if torch.is_tensor(raw_predictions) else np.asarray(raw_predictions)
    predictions = np.squeeze(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    expected_horizon = config['pipeline']['decoder_window']
    actual_df = test_df.tail(expected_horizon).reset_index(drop=True)
    actual = actual_df['load_mw'].to_numpy(dtype=float)

    if predictions.shape[0] != len(actual):
        raise RuntimeError(
            f"Prediction horizon mismatch: predicted {predictions.shape[0]} steps, actual {len(actual)} steps."
        )

    p10 = predictions[:, 0] if predictions.shape[1] > 0 else predictions[:, 0]
    p50 = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
    p90 = predictions[:, 2] if predictions.shape[1] > 2 else predictions[:, -1]

    metrics = _metrics_from_predictions(actual, p50)
    metrics.update({
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "checkpoint_epoch_loss": float(checkpoint.get("callbacks", {}).get("ModelCheckpoint", {}).get("best_model_score", float('nan')))
        if isinstance(checkpoint.get("callbacks", {}), dict) else float('nan'),
    })

    compare_df = actual_df[['timestamp', 'load_mw']].copy()
    compare_df = compare_df.rename(columns={'load_mw': 'actual_load_mw'})
    compare_df['p10'] = p10
    compare_df['p50'] = p50
    compare_df['p90'] = p90
    compare_df['error_p50'] = compare_df['actual_load_mw'] - compare_df['p50']
    compare_df['abs_error_p50'] = np.abs(compare_df['error_p50'])
    compare_df['ape_p50'] = np.abs(compare_df['error_p50'] / np.clip(compare_df['actual_load_mw'], 1e-6, None)) * 100

    testing_root = Path(config['data']['models_root']) / 'testing'
    testing_root.mkdir(parents=True, exist_ok=True)
    output_dir = testing_root / f"{run_id}_epoch1_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_path = output_dir / 'test_predictions_vs_actual.csv'
    metrics_path = output_dir / 'metrics.json'
    compare_df.to_csv(compare_path, index=False)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"[TEST] Saved predictions vs actual to {compare_path}")
    print(f"[TEST] Saved metrics to {metrics_path}")
    print("[TEST] Metrics:")
    for key in ["MAE", "RMSE", "MAPE", "SMAPE"]:
        print(f"  {key}: {metrics[key]:.4f}")

    return {
        "metrics": metrics,
        "comparison_path": str(compare_path),
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    run_test_pipeline()
