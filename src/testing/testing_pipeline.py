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
from pytorch_forecasting import TemporalFusionTransformer
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


def _parse_val_loss_from_filename(path: Path) -> float:
    """
    Extract the val_loss value embedded in a checkpoint filename.

    Handles both naming styles:
      - epoch=08-val_loss=80.13.ckpt          (clean)
      - epoch=epoch=08-val_loss=val_loss=80.13.ckpt  (Lightning double-prefix)
    Returns float('inf') if no val_loss is found so the file sorts last.
    """
    name = path.stem  # strip .ckpt
    if 'val_loss=' not in name:
        return float('inf')
    try:
        # Take the part after the LAST 'val_loss=' — handles double-prefix correctly
        return float(name.split('val_loss=')[-1])
    except ValueError:
        return float('inf')


def _find_best_checkpoint_globally(config: dict[str, Any]) -> Path:
    """
    Scan ALL run folders under models/runs/ and return the single checkpoint
    with the lowest val_loss embedded in its filename.

    Falls back to the most-recent last.ckpt if no val_loss-named file exists.
    """
    checkpoints_dir = (
        Path(config['data']['models_root'])
        / config['data']['models_checkpoints_dir'].replace('models/', '').rstrip('/')
    )
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {checkpoints_dir}")

    # Gather every epoch=*.ckpt across all run sub-folders
    all_ckpts = list(checkpoints_dir.rglob('epoch=*.ckpt'))

    if not all_ckpts:
        # No named checkpoints at all — fall back to most-recent last.ckpt
        last_ckpts = sorted(
            checkpoints_dir.rglob('last.ckpt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if last_ckpts:
            print(f"[TEST] No epoch=*.ckpt files found; falling back to: {last_ckpts[0]}")
            return last_ckpts[0]
        raise FileNotFoundError(f"No checkpoints found anywhere under {checkpoints_dir}")

    # Sort by parsed val_loss ascending — lowest = best
    all_ckpts.sort(key=_parse_val_loss_from_filename)
    best = all_ckpts[0]
    best_loss = _parse_val_loss_from_filename(best)

    print(f"[TEST] Scanned {len(all_ckpts)} checkpoint(s) across all runs.")
    print(f"[TEST] Best checkpoint: {best.name}  (val_loss={best_loss:.4f})")
    return best


def _select_checkpoint(config: dict[str, Any], run_id: str, checkpoint_name: str | None = None) -> Path:
    """Pick the best checkpoint from a specific run folder."""
    checkpoints_dir = (
        Path(config['data']['models_root'])
        / config['data']['models_checkpoints_dir'].replace('models/', '').rstrip('/')
    )
    run_dir = checkpoints_dir / run_id

    if checkpoint_name is not None:
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidate_files = list(run_dir.glob('epoch=*.ckpt'))
    if not candidate_files:
        last_ckpt = run_dir / 'last.ckpt'
        if last_ckpt.exists():
            return last_ckpt
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    # Sort by val_loss ascending — pick the lowest
    candidate_files.sort(key=_parse_val_loss_from_filename)
    best = candidate_files[0]
    print(f"[TEST] Best in run '{run_id}': {best.name}  (val_loss={_parse_val_loss_from_filename(best):.4f})")
    return best


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


def run_test_pipeline(
    config_path: str = "config/config.yaml",
    run_id: str | None = None,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate the best checkpoint on the held-out test horizon and persist outputs.

    Checkpoint selection priority:
      1. checkpoint_name given  →  load that exact file from run_id's folder
      2. run_id given           →  pick the lowest val_loss file in that run's folder
      3. neither given          →  scan ALL runs and pick the globally lowest val_loss
    """
    config = load_config(config_path)
    _register_safe_globals_for_checkpoint_resume()

    if run_id is not None:
        print(f"[TEST] Targeting run: {run_id}")
        checkpoint_path = _select_checkpoint(config, run_id, checkpoint_name)
    else:
        print("[TEST] No run_id specified — scanning all runs for global best checkpoint...")
        checkpoint_path = _find_best_checkpoint_globally(config)
        # Infer run_id from the checkpoint's parent directory name
        run_id = checkpoint_path.parent.name

    print(f"[TEST] Checkpoint: {checkpoint_path}")

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
    # Use the actual run_id + checkpoint stem so output folders never collide
    output_dir = testing_root / f"{run_id}_{checkpoint_path.stem}"
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
