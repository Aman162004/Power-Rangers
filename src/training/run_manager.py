"""Training run management: checkpoint tracking, run metadata, and artifact organization."""

import os
import json
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
