"""Forecast engine: loads the active trained model and generates predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import yaml

import numpy as np
import pandas as pd


class ForecastEngine:
    """Inference engine that loads the active trained checkpoint."""

    def __init__(self, config_path: str):
        """Initialize with config to resolve active model path."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Placeholder: loads active checkpoint path (not actual model yet)
        self.model = self._load_active_model()

    def _load_active_model(self) -> Optional[dict]:
        """Load the currently active trained model checkpoint."""
        models_root = Path(self.config['data']['models_root'])
        active_pointer = models_root / self.config['data']['active_model_pointer'].replace('models/', '')
        
        if not active_pointer.exists():
            print("[WARN] No active model pointer found. Using placeholder model.")
            return None
        
        with open(active_pointer, 'r') as f:
            run_id = f.read().strip()
        
        checkpoints_dir = Path(self.config['data']['models_checkpoints_dir'])
        best_ckpt = checkpoints_dir / run_id / 'best.ckpt'
        
        if not best_ckpt.exists():
            print(f"[WARN] Checkpoint not found at {best_ckpt}. Using placeholder model.")
            return None
        
        return {
            "model_name": f"tft_model_{run_id}",
            "checkpoint_path": str(best_ckpt),
            "run_id": run_id,
            "exists": True,
        }

    def generate_forecast(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic forecast.
        
        Placeholder implementation; returns dummy predictions.
        Real implementation will:
        1. Load TFT checkpoint
        2. Prepare input tensors
        3. Generate P10, P50, P90 quantile forecasts
        """
        if "timestamp" not in input_data.columns:
            timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=len(input_data), freq="15min")
        else:
            timestamps = pd.to_datetime(input_data["timestamp"])

        # Placeholder: linear extrapolation
        if "load_mw" in input_data.columns:
            last_load = input_data["load_mw"].iloc[-1]
            base_val = float(last_load)
        else:
            base_val = 3000.0
        
        forecast_steps = 96  # 24 hours at 15-min resolution
        forecast_timestamps = pd.date_range(start=timestamps.iloc[-1], periods=forecast_steps + 1, freq="15min")[1:]
        
        p50 = np.linspace(base_val - 100, base_val + 100, forecast_steps)
        p10 = p50 - 150
        p90 = p50 + 150

        return pd.DataFrame({
            "timestamp": forecast_timestamps,
            "p10": p10,
            "p50": p50,
            "p90": p90,
        })

