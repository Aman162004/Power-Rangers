"""Training pipeline: loads splits, trains TFT model, manages checkpoints under models/."""

import re
import os
import sys
import warnings
import yaml
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pytorch_forecasting uses sklearn scalers internally and can emit this warning
# repeatedly during batch transforms; it is noisy but not actionable here.
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    module=r"sklearn\.utils\.validation",
)

from src.training.run_manager import TrainingRunManager, find_latest_resumable_run
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def _register_safe_globals_for_checkpoint_resume() -> None:
    """
    Allowlist trusted pytorch_forecasting classes for torch.load(weights_only=True).

    PyTorch 2.6 defaults to weights_only=True. Lightning resume can fail unless
    these encoder/normalizer classes are registered as safe globals.
    """
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

    # NumPy scalar/dtype globals encountered in Lightning checkpoints.
    # Use numpy._core when available to avoid deprecated numpy.core path.
    np_core = getattr(np, "_core", None)
    if np_core is None:
        np_core = np.core
    numpy_scalar_global = np_core.multiarray.scalar
    numpy_dtype_base = np.dtype
    numpy_float64_dtype_cls = type(np.dtype("float64"))
    numpy_float32_dtype_cls = type(np.dtype("float32"))

    add_safe_globals(
        [
            EncoderNormalizer,
            GroupNormalizer,
            MultiNormalizer,
            NaNLabelEncoder,
            TorchNormalizer,
            numpy_scalar_global,
            numpy_dtype_base,
            numpy_float64_dtype_cls,
            numpy_float32_dtype_cls,
        ]
    )


def load_config(config_path: str) -> dict:
    """Load config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_training_splits(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training/val/test splits from data/historical/final_processed/.
    Fail fast if any split is missing; don't fall back to re-preprocessing.
    """
    splits_path = Path(config['data']['historical_splits_path'])
    if not splits_path.is_absolute():
        splits_path = PROJECT_ROOT / splits_path
    
    split_files = {
        'train': splits_path / config['data']['training_splits']['train'],
        'val': splits_path / config['data']['training_splits']['val'],
        'test': splits_path / config['data']['training_splits']['test'],
    }
    
    # Validate all splits exist
    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(
                f"Training split '{split_name}' not found at {split_file}. "
                f"Run data preparation pipeline first."
            )
    
    train_df = pd.read_parquet(split_files['train'])
    val_df = pd.read_parquet(split_files['val'])
    test_df = pd.read_parquet(split_files['test'])
    
    return train_df, val_df, test_df


def _drop_unused_training_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop audit columns and redundant features that add no value to training.
    Keeps only: target, exogenous, time features, lags, rolling stats, required TFT columns.
    """
    df = df.copy()
    
    # Explicit columns to drop
    drop_cols = config['data'].get('training_drop_columns', [])
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    # Pattern-based drops (e.g., *_was_missing)
    drop_pattern = config['data'].get('training_drop_pattern', '')
    if drop_pattern:
        pattern_cols = [col for col in df.columns if re.match(drop_pattern, col)]
        drop_cols.extend(pattern_cols)
    
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)), errors='ignore')
    
    return df


def _validate_required_columns(df: pd.DataFrame, split_name: str) -> None:
    """Ensure all required columns for TFT training are present."""
    required = {
        'load_mw': 'target',
        'time_idx': 'TFT required',
        'group_id': 'TFT required',
        'hour': 'time feature',
        'day_of_week': 'time feature',
        'month': 'time feature',
        'sin_hour': 'cyclical encoding',
        'cos_hour': 'cyclical encoding',
        'is_holiday': 'calendar feature',
        'temperature': 'weather',
        'humidity': 'weather',
        'wind_speed': 'weather',
        'rainfall': 'weather',
    }
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {split_name} split: {missing}. "
            f"Ensure data preparation pipeline has run and includes all features."
        )


def _build_tft_datasets(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Build pytorch_forecasting TimeSeriesDataSet objects for TFT."""
    lag_cols = [f'load_lag_{lag}' for lag in config['features']['lags']]
    rolling_cols = [f'rolling_mean_{w}' for w in config['features']['rolling_windows']]
    
    # Validate all required columns present
    _validate_required_columns(train_df, 'train')
    _validate_required_columns(val_df, 'val')
    _validate_required_columns(test_df, 'test')
    
    max_encoder_length = config['pipeline']['encoder_window']
    max_prediction_length = config['pipeline']['decoder_window']
    
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="load_mw",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['hour', 'day_of_week', 'month', 'sin_hour', 'cos_hour', 'temperature', 'humidity', 'wind_speed', 'rainfall', 'is_holiday'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['load_mw'] + lag_cols + rolling_cols,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    val_dataset = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
    
    return training, val_dataset, test_dataset


def run_training_pipeline(config_path: str = "config/config.yaml", run_id: str = None):
    """
    Main training pipeline:
    1. Load prepared splits from data/historical/final_processed/
    2. Drop unused columns (audit, redundant)
    3. Validate required features
    4. Build TFT datasets
    5. Train with Lightning, resuming from last checkpoint if policy allows
    6. Save all artifacts under models/runs/<run_id>/
    """
    config = load_config(config_path)

    # Ensure checkpoints from trusted local runs can be resumed under PyTorch>=2.6.
    _register_safe_globals_for_checkpoint_resume()

    # Better Tensor Core utilization on compatible NVIDIA GPUs.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        if config.get("training", {}).get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
    
    resume_policy = config['training'].get('resume_policy', 'auto')

    # Choose run folder: resume latest existing run for auto/require unless explicit run_id is passed.
    resume_ckpt_path = None
    resolved_run_id = run_id
    if resolved_run_id is None and resume_policy in {'auto', 'require'}:
        latest_run_id, latest_ckpt = find_latest_resumable_run(config)
        if latest_run_id is not None and latest_ckpt is not None:
            resolved_run_id = latest_run_id
            resume_ckpt_path = str(latest_ckpt)
            print(f"[RUN] Resuming latest run: {resolved_run_id}")
        elif resume_policy == 'require':
            raise RuntimeError("Resume policy is 'require' but no previous run with last.ckpt was found.")

    # Initialize run manager
    run_manager = TrainingRunManager(config, run_id=resolved_run_id)
    if resume_ckpt_path is None:
        print(f"[RUN] Starting training run: {run_manager.run_id}")
    
    # Save config snapshot
    config_snapshot_path = run_manager.save_config_snapshot()
    print(f"[CONFIG] Snapshot saved to {config_snapshot_path}")
    
    # Load splits directly from prepared data
    print("[DATA] Loading training splits...")
    train_df, val_df, test_df = _load_training_splits(config)
    print(f"[DATA] Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Drop unused columns
    print("[DATA] Dropping audit and redundant columns...")
    train_df = _drop_unused_training_columns(train_df, config)
    val_df = _drop_unused_training_columns(val_df, config)
    test_df = _drop_unused_training_columns(test_df, config)
    print(f"[DATA] After cleanup: {len(train_df.columns)} features remaining")
    
    # Validate split sizes
    min_required_train_rows = config['pipeline']['encoder_window'] + config['pipeline']['decoder_window']
    if len(train_df) < min_required_train_rows or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Insufficient split sizes for TFT training: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
            f"required_train>={min_required_train_rows}."
        )
    
    # Build TFT datasets
    print("[DATASET] Building TFT datasets...")
    train_dataset, val_dataset, test_dataset = _build_tft_datasets(config, train_df, val_df, test_df)
    
    # Create dataloaders tuned for throughput.
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

    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=config['model']['batch_size'],
        **dataloader_kwargs,
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=config['model']['batch_size'],
        **dataloader_kwargs,
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=config['model']['batch_size'],
        **dataloader_kwargs,
    )
    
    # Initialize model
    print("[MODEL] Initializing TFT model...")
    precision_mode = str(config.get('training', {}).get('precision', '32-true'))
    # In fp16, very large negative mask bias (e.g. -1e9) can overflow to half.
    # Use a safe magnitude while retaining masking behavior.
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
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config['training']['early_stopping_patience']
    )
    checkpoint_strategy = config.get('training', {}).get('checkpoint_save_strategy', 'best')
    checkpoint_every_n_epochs = int(config.get('training', {}).get('checkpoint_every_n_epochs', 1))

    if checkpoint_strategy == 'all':
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_manager.run_checkpoints_dir),
            filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',
            save_top_k=-1,
            monitor='val_loss',
            save_last=True,
            every_n_epochs=checkpoint_every_n_epochs,
        )
    elif checkpoint_strategy == 'last':
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_manager.run_checkpoints_dir),
            filename='last',
            save_top_k=0,
            save_last=True,
        )
    else:
        # Default: keep best + last
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_manager.run_checkpoints_dir),
            filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            save_last=True,
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        precision=config.get('training', {}).get('precision', '32-true'),
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        devices=1 if torch.cuda.is_available() else 'auto',
        benchmark=bool(config.get('training', {}).get('cudnn_benchmark', True)),
    )
    
    # Resolve checkpoint to resume from
    ckpt_path = resume_ckpt_path

    if resume_policy == 'scratch':
        print("[CHECKPOINT] Training from scratch")
        ckpt_path = None
    elif ckpt_path is not None:
        print(f"[CHECKPOINT] Resuming from {ckpt_path}")
    elif resume_policy in {'auto', 'require'}:
        # Fallback for explicitly requested run_id.
        last_ckpt = run_manager.get_last_checkpoint()
        if last_ckpt:
            ckpt_path = str(last_ckpt)
            print(f"[CHECKPOINT] Resuming from {ckpt_path}")
        elif resume_policy == 'require':
            raise RuntimeError("Resume policy is 'require' but no checkpoint found for selected run.")
    
    # Train
    print("[TRAINING] Starting training...")
    resume_weights_only = bool(config.get('training', {}).get('resume_weights_only', False))
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=ckpt_path,
        weights_only=resume_weights_only,
    )
    
    # Test
    print("[TESTING] Running evaluation on test set...")
    trainer.test(model, test_dataloader)
    
    # Update metadata
    run_manager.update_metadata('best_val_loss', checkpoint_callback.best_model_score.item())
    run_manager.update_metadata('epochs_completed', trainer.current_epoch)
    run_manager.finalize(status='completed')
    
    print(f"[COMPLETE] Training completed. Artifacts saved to {run_manager.run_checkpoints_dir}")
    print(f"[METADATA] Run metadata: {run_manager.metadata}")


if __name__ == "__main__":
    config_path = "config/config.yaml"
    run_training_pipeline(config_path)
