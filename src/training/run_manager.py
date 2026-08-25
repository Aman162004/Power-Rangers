"""Training run management: checkpoint tracking, run metadata, and artifact organization."""

import os
import json
import warnings
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Tuple


class TrainingRunManager:
    """Manages a single training run lifecycle: init, checkpoint save/load, metadata."""

    def __init__(self, config: Dict[str, Any], run_id: Optional[str] = None):
        """
        Initialize run manager.

        Args:
            config: Configuration dict with models_root, models_checkpoints_dir, etc.
            run_id: Unique run identifier. If None, generates one from timestamp.
        """
        self.config = config
        self.run_id = run_id or self._generate_run_id()
        self.models_root = Path(config['data']['models_root'])
        self.checkpoints_dir = self.models_root / config['data']['models_checkpoints_dir'].replace('models/', '').rstrip('/')
        self.config_dir = self.models_root / config['data']['models_config_dir'].replace('models/', '').rstrip('/')
        
        # Per-run paths
        self.run_checkpoints_dir = self.checkpoints_dir / self.run_id
        self.run_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata = {
            'run_id': self.run_id,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'initialized',
            'epochs_completed': 0,
            'best_val_loss': float('inf'),
        }

    def _generate_run_id(self) -> str:
        """Generate unique run ID from timestamp."""
        return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    def get_checkpoint_path(self, name: str = 'last') -> Path:
        """Get path to a named checkpoint."""
        return self.run_checkpoints_dir / f'{name}.ckpt'

    def get_last_checkpoint(self) -> Optional[Path]:
        """Return path to last checkpoint if it exists, else None."""
        last_ckpt = self.get_checkpoint_path('last')
        return last_ckpt if last_ckpt.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Return path to best checkpoint if it exists, else None."""
        best_ckpt = self.get_checkpoint_path('best')
        return best_ckpt if best_ckpt.exists() else None

    def save_config_snapshot(self) -> Path:
        """Save config snapshot to models/config/<run_id>.yaml."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = self.config_dir / f'{self.run_id}.yaml'
        if config_snapshot_path.exists():
            return config_snapshot_path
        with open(config_snapshot_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        return config_snapshot_path

    def save_metadata(self) -> Path:
        """Save run metadata to <run_checkpoints_dir>/metadata.json."""
        metadata_path = self.run_checkpoints_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        return metadata_path

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata field."""
        self.metadata[key] = value

    def finalize(self, status: str = 'completed') -> None:
        """Mark run as completed and save metadata."""
        self.metadata['status'] = status
        self.metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        self.save_metadata()

    def set_as_active_model(self) -> None:
        """Mark this run's best model as the active/deployed model."""
        active_pointer_path = self.models_root / self.config['data']['active_model_pointer'].replace('models/', '')
        active_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(active_pointer_path, 'w') as f:
            f.write(self.run_id)


def load_active_run_id(models_root: Path) -> Optional[str]:
    """Load the active run ID from ACTIVE_MODEL.txt, or None if not set."""
    pointer_path = models_root / 'ACTIVE_MODEL.txt'
    if pointer_path.exists():
        return pointer_path.read_text().strip()
    return None


def resolve_active_checkpoint(models_root: Path, models_checkpoints_dir: str) -> Optional[Path]:
    """Resolve the path to the currently active best checkpoint."""
    run_id = load_active_run_id(models_root)
    if not run_id:
        return None
    checkpoints_dir = Path(models_checkpoints_dir.replace('models/', models_root / ''))
    best_ckpt = checkpoints_dir / run_id / 'best.ckpt'
    return best_ckpt if best_ckpt.exists() else None


def find_latest_resumable_run(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Path]]:
    """
    Return the most recent run id that contains last.ckpt.

    Runs are named as UTC timestamps (YYYYMMDD_HHMMSS), so lexical sort is chronological.
    """
    models_root = Path(config['data']['models_root'])
    checkpoints_dir = models_root / config['data']['models_checkpoints_dir'].replace('models/', '').rstrip('/')
    if not checkpoints_dir.exists():
        return None, None

    run_dirs = [p for p in checkpoints_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None, None

    for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        last_ckpt = run_dir / 'last.ckpt'
        if last_ckpt.exists():
            return run_dir.name, last_ckpt

    return None, None


def _read_geometry_snapshot(checkpoint_path: Path, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the training-time config from the cheapest source available.

    Order:
      1. `models/config/<run_id>.yaml` — written by `TrainingRunManager.save_config_snapshot`.
         No torch required.
      2. The .ckpt file itself via `torch.load(weights_only=False)`. Reads
         `dataset_parameters` from hparams (pytorch_forecasting layout).

    Returns the geometry-bearing subset, or None if neither source is reachable.
    """
    try:
        run_id = checkpoint_path.parent.name
        models_root = Path(config['data']['models_root'])
        models_config_dir_rel = config['data']['models_config_dir'].replace('models/', '').rstrip('/')
        snapshot_path = models_root / models_config_dir_rel / f'{run_id}.yaml'
        if snapshot_path.exists():
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snap = yaml.safe_load(f) or {}
            return {
                'pipeline': snap.get('pipeline', {}),
                'features': snap.get('features', {}),
            }
    except Exception as exc:
        warnings.warn(f"validate_checkpoint_geometry: snapshot yaml read failed: {exc}")

    try:
        import torch
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        hp = ckpt.get('hyper_parameters', {}) if isinstance(ckpt, dict) else {}
        ds_params = hp.get('dataset_parameters', {})
        if not ds_params:
            ds_params = ckpt.get('dataset_parameters', {}) if isinstance(ckpt, dict) else {}
        return {
            'pipeline': {
                'encoder_window': ds_params.get('max_encoder_length'),
                'decoder_window': ds_params.get('max_prediction_length'),
            },
            'features': {
                'lags': [int(c.split('_')[-1]) for c in hp.get('time_varying_unknown_reals', [])
                         if isinstance(c, str) and c.startswith('load_lag_')],
                'rolling_windows': [int(c.split('_')[-1]) for c in hp.get('time_varying_unknown_reals', [])
                                    if isinstance(c, str) and c.startswith('rolling_mean_')],
            },
        }
    except Exception as exc:
        warnings.warn(f"validate_checkpoint_geometry: torch.load failed: {exc}")


def validate_checkpoint_geometry(checkpoint_path, config):
    """Raise a clear, actionable error if a checkpoint's geometry does not match the active config.

    Compares (in order of severity):
      1. encoder_window — mismatches produce silent tensor-shape errors inside
         the LSTM / variable-selection networks during forward pass.
      2. decoder_window — same.
      3. lags, rolling_windows — mismatches produce silent column-order /
         feature-count errors inside TimeSeriesDataSet.from_dataset.

    Designed to be cheap on the happy path: the snapshot yaml is just a file
    read; only falls back to torch.load on the rare case where the snapshot
    is missing (which happens for manually-promoted checkpoints like
    `models/final model/`).

    Called from:
      - `src/forecast/tft_inference.py::_load_tft_model` (serving path)
      - `src/training/training_pipeline.py::run_training_pipeline` (resume flow)

    On raise, the caller must NOT proceed.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"validate_checkpoint_geometry: checkpoint not found at {checkpoint_path}")

    snapshot = _read_geometry_snapshot(checkpoint_path, config)
    if snapshot is None:
        warnings.warn(
            f"validate_checkpoint_geometry: cannot read geometry from {checkpoint_path} "
            "(snapshot missing, torch.load failed). Proceeding without geometry check — "
            "this is dangerous. Restore the snapshot yaml or rebuild the checkpoint.",
            stacklevel=2,
        )
        return

    active_pipeline = config.get('pipeline', {})
    active_features = config.get('features', {})
    snap_pipeline = snapshot.get('pipeline', {})
    snap_features = snapshot.get('features', {})

    expected_enc = active_pipeline.get('encoder_window')
    actual_enc = snap_pipeline.get('encoder_window')
    expected_dec = active_pipeline.get('decoder_window')
    actual_dec = snap_pipeline.get('decoder_window')
    expected_lags = sorted(active_features.get('lags', []) or [])
    actual_lags = sorted(snap_features.get('lags', []) or [])
    expected_roll = sorted(active_features.get('rolling_windows', []) or [])
    actual_roll = sorted(snap_features.get('rolling_windows', []) or [])

    errors = []
    if actual_enc is not None and expected_enc is not None and int(actual_enc) != int(expected_enc):
        errors.append(f"encoder_window: checkpoint={actual_enc}, config={expected_enc}")
    if actual_dec is not None and expected_dec is not None and int(actual_dec) != int(expected_dec):
        errors.append(f"decoder_window: checkpoint={actual_dec}, config={expected_dec}")
    if expected_lags and actual_lags and expected_lags != actual_lags:
        errors.append(f"lags: checkpoint={actual_lags}, config={expected_lags}")
    if expected_roll and actual_roll and expected_roll != actual_roll:
        errors.append(f"rolling_windows: checkpoint={actual_roll}, config={expected_roll}")

    if errors:
        raise RuntimeError(
            "\n[validate_checkpoint_geometry] CHECKPOINT GEOMETRY MISMATCH\n"
            f"  Checkpoint: {checkpoint_path}\n"
            f"  Active config: encoder_window={expected_enc}, decoder_window={expected_dec}, "
            f"lags={expected_lags}, rolling_windows={expected_roll}\n"
            f"  Mismatches:\n    - " + "\n    - ".join(errors) + "\n\n"
            "  Refusing to load incompatible checkpoint. Either:\n"
            "    a) retrain under the active config, OR\n"
            "    b) revert the active config to match this checkpoint, OR\n"
            "    c) archive this checkpoint under models/_archive/ (see Step 5 notes)."
        )
