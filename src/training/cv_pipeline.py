"""Cross-validation pipeline for tuning TFT hyperparameters."""

import os
import gc
import sys
import json
import yaml
import torch
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
from datetime import datetime, timezone
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import required functions from training pipeline
from src.training.training_pipeline import (
    load_config,
    _drop_unused_training_columns,
    _validate_required_columns,
    LRWarmupCallback,
    _register_safe_globals_for_checkpoint_resume
)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def build_datasets_for_cv(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Build dataset for a specific CV split."""
    lag_cols = [f'load_lag_{lag}' for lag in config['features']['lags']]
    rolling_cols = [f'rolling_mean_{w}' for w in config['features']['rolling_windows']]

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
    return training, val_dataset

def run_cv_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    _register_safe_globals_for_checkpoint_resume()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        if config.get("training", {}).get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True

    # 1. Load the continuous cleaned history (train + val + test)
    # The safest way is to load train and val and combine them
    splits_path = Path(config['data']['historical_splits_path'])
    train_file = splits_path / config['data']['training_splits']['train']
    val_file = splits_path / config['data']['training_splits']['val']

    print(f"[DATA] Loading data for CV from {train_file} and {val_file}")
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    
    # Combine train and val into single continuous dataframe for TimeSeriesSplit
    combined_df = pd.concat([train_df, val_df]).sort_values('time_idx').reset_index(drop=True)
    
    # [CV SPEED] Take only the last 20,000 rows (approx 7 months) for CV tuning.
    # This is plenty of data to compare hidden_size 64 vs 128 without training for days.
    if len(combined_df) > 20000:
        print(f"[CV SPEED] Subsetting data from {len(combined_df)} to last 20,000 rows.")
        combined_df = combined_df.tail(20000).reset_index(drop=True)

    combined_df = _drop_unused_training_columns(combined_df, config)

    n_splits = config.get('training', {}).get('cv_tuning', {}).get('n_splits', 3)
    hidden_sizes = config.get('training', {}).get('cv_tuning', {}).get('grid', {}).get('hidden_size', [64])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {}
    best_hidden_size = None
    best_mean_val_loss = float('inf')

    print(f"\n[CV START] Running {n_splits}-fold TimeSeriesSplit CV")
    print(f"Grid: hidden_size = {hidden_sizes}")
    
    for hs in hidden_sizes:
        print(f"\n{'='*50}\nTesting hidden_size = {hs}\n{'='*50}")
        fold_losses = []
        
        # Override config for this run
        config['model']['hidden_size'] = hs
        
        for fold, (train_index, val_index) in enumerate(tscv.split(combined_df)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            fold_train_df = combined_df.iloc[train_index].copy()
            fold_val_df = combined_df.iloc[val_index].copy()

            # Ensure minimum sizes
            min_required = config['pipeline']['encoder_window'] + config['pipeline']['decoder_window']
            if len(fold_train_df) < min_required or len(fold_val_df) < min_required:
                print(f"Skipping fold {fold+1}: Not enough data")
                continue

            train_ds, val_ds = build_datasets_for_cv(config, fold_train_df, fold_val_df)
            
            # Setup Dataloaders
            num_workers = int(config.get('training', {}).get('num_workers', 0))
            dataloader_kwargs = {
                'num_workers': num_workers,
                'pin_memory': bool(config.get('training', {}).get('pin_memory', True)),
            }
            if num_workers > 0:
                dataloader_kwargs['persistent_workers'] = bool(config.get('training', {}).get('persistent_workers', True))
                dataloader_kwargs['prefetch_factor'] = int(config.get('training', {}).get('prefetch_factor', 4))

            train_dl = train_ds.to_dataloader(train=True, batch_size=config['model']['batch_size'], **dataloader_kwargs)
            val_dl = val_ds.to_dataloader(train=False, batch_size=config['model']['batch_size'], **dataloader_kwargs)

            precision_mode = str(config.get('training', {}).get('precision', '32-true'))
            mask_bias = -1.0e4 if '16' in precision_mode else -1.0e9

            model = TemporalFusionTransformer.from_dataset(
                train_ds,
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

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=3, # Reduced patience for CV speed
                mode='min',
            )

            warmup_cfg = config.get('training', {}).get('lr_warmup', {})
            lr_warmup_callback = LRWarmupCallback(
                warmup_epochs=int(warmup_cfg.get('epochs', 5)),
                start_lr=float(warmup_cfg.get('start_lr', 1e-5)),
                target_lr=float(config['model']['learning_rate']),
            )

            trainer = pl.Trainer(
                max_epochs=config['model']['max_epochs'],
                callbacks=[early_stopping, lr_warmup_callback],
                enable_progress_bar=True,
                logger=False, # Disable massive logs for CV
                precision=config.get('training', {}).get('precision', '32-true'),
                accelerator='gpu' if torch.cuda.is_available() else 'auto',
                devices=1 if torch.cuda.is_available() else 'auto',
                benchmark=bool(config.get('training', {}).get('cudnn_benchmark', True)),
                gradient_clip_val=float(config.get('training', {}).get('gradient_clip_val', 0.1)),
                gradient_clip_algorithm=config.get('training', {}).get('gradient_clip_algorithm', 'norm'),
            )

            trainer.fit(model, train_dl, val_dl)
            
            # Record best loss for this fold
            best_val_loss = early_stopping.best_score.item() if early_stopping.best_score is not None else float('inf')
            fold_losses.append(best_val_loss)
            print(f"Fold {fold+1} validation loss: {best_val_loss:.4f}")

            # [MEMORY CLEANUP] Prevent GPU memory from leaking across CV folds
            del model
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if fold_losses:
            mean_loss = sum(fold_losses) / len(fold_losses)
            results[hs] = {'mean_loss': mean_loss, 'fold_losses': fold_losses}
            print(f"-> Mean validation loss for hidden_size={hs}: {mean_loss:.4f}")
            
            if mean_loss < best_mean_val_loss:
                best_mean_val_loss = mean_loss
                best_hidden_size = hs

    print(f"\n[CV COMPLETE] Best hidden_size: {best_hidden_size} (Mean Loss: {best_mean_val_loss:.4f})")
    
    cv_report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'best_hidden_size': best_hidden_size,
        'best_mean_val_loss': best_mean_val_loss,
        'grid_results': results
    }

    report_path = Path(config['data']['models_checkpoints_dir']) / "cv_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(cv_report, f, indent=2)
    print(f"Results saved to {report_path}")

if __name__ == "__main__":
    run_cv_pipeline()
