"""Scaffold inference utilities for the forecasting system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for scaffold mode
    yaml = None


def load_model(checkpoint_path: str) -> dict[str, Any]:
    """Load a placeholder model reference from disk.

    Args:
        checkpoint_path: Path to a model checkpoint file.

    Returns:
        Lightweight metadata describing the requested checkpoint.
    """

    checkpoint = Path(checkpoint_path)
    return {
        "model_name": "placeholder_forecast_model",
        "checkpoint_path": str(checkpoint),
        "exists": checkpoint.exists(),
    }


def load_test_data(path: str) -> pd.DataFrame:
    """Load test data for inference, falling back to dummy data when needed.

    Args:
        path: Path to a CSV or parquet file containing test rows.

    Returns:
        A dataframe with timestamps and optional load columns.
    """

    data_path = Path(path)
    if data_path.exists():
        try:
            if data_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception:
            pass

    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=24, freq="15min")
    load_values = np.linspace(3000.0, 3200.0, num=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "load_mw": load_values, "actual": load_values})


def run_inference(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Run placeholder inference and return predictions aligned with timestamps.

    Args:
        model: Placeholder model metadata.
        data: Input rows with an optional timestamp column.

    Returns:
        A dataframe containing timestamps and placeholder forecast outputs.
    """

    _ = model
    prepared = _ensure_input_frame(data)

    if prepared.empty:
        return pd.DataFrame(columns=["timestamp", "prediction", "p10", "p50", "p90", "model_name"])

    prediction = _build_placeholder_predictions(prepared)
    p10 = prediction - 150.0
    p90 = prediction + 150.0

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(prepared["timestamp"]),
            "prediction": prediction,
            "p10": p10,
            "p50": prediction,
            "p90": p90,
            "model_name": str(model.get("model_name", "unknown_model")),
        }
    )


def _ensure_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized input dataframe for placeholder inference."""

    prepared = data.copy()
    if "timestamp" not in prepared.columns:
        prepared["timestamp"] = pd.date_range(start="2026-01-01 00:00:00", periods=len(prepared), freq="15min")
    else:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    return prepared


def _build_placeholder_predictions(data: pd.DataFrame) -> np.ndarray:
    """Create simple placeholder predictions using available load history."""

    if "load_mw" in data.columns:
        baseline = pd.to_numeric(data["load_mw"], errors="coerce").ffill().fillna(3000.0)
        values = baseline.to_numpy(dtype=float)
    elif "actual" in data.columns:
        baseline = pd.to_numeric(data["actual"], errors="coerce").ffill().fillna(3000.0)
        values = baseline.to_numpy(dtype=float)
    else:
        values = np.linspace(3000.0, 3200.0, num=max(len(data), 1), dtype=float)

    if len(values) == 0:
        return np.array([], dtype=float)

    return values


class ForecastEngine:
    """Project-facing inference engine backed by placeholder logic."""

    def __init__(self, config_path: str):
        """Initialize the engine from the project configuration."""

        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = load_model(self._resolve_checkpoint_path())

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load YAML configuration when the optional parser is available."""

        if yaml is None:
            return {}

        config_file = Path(config_path)
        if not config_file.exists():
            return {}

        with config_file.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        return loaded or {}

    def _resolve_checkpoint_path(self) -> str:
        """Resolve the preferred checkpoint path from current project artifacts."""

        project_root = Path(self.config_path).resolve().parents[1]
        candidate_paths = [
            project_root / "models" / "final model",
            project_root / "models" / "runs",
        ]

        checkpoint_files: list[Path] = []
        for candidate in candidate_paths:
            if candidate.is_file() and candidate.suffix == ".ckpt":
                checkpoint_files.append(candidate)
            elif candidate.exists():
                checkpoint_files.extend(sorted(candidate.rglob("*.ckpt")))

        if checkpoint_files:
            return str(checkpoint_files[0])

        return str(project_root / "models" / "placeholder_model.ckpt")

    def generate_forecast(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Generate a placeholder probabilistic forecast for the provided rows."""

        prepared = _ensure_input_frame(input_data)
        return run_inference(self.model, prepared)
