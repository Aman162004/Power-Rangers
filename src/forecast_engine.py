"""Placeholder inference utilities for the forecasting system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_model(checkpoint_path: str) -> dict[str, Any]:
    """Load a forecasting model from disk.

    This is a scaffold implementation that returns lightweight metadata instead
    of reconstructing a real model object.

    Args:
        checkpoint_path: Path to the serialized model checkpoint.

    Returns:
        A placeholder model representation that downstream code can consume.
    """

    return {
        "model_name": "placeholder_forecast_model",
        "checkpoint_path": checkpoint_path,
        "exists": Path(checkpoint_path).exists(),
    }


def load_test_data(path: str) -> pd.DataFrame:
    """Load inference-time test data.

    If the provided file does not exist, the function returns a dummy dataframe
    so the scaffold can run end-to-end without external setup.

    Args:
        path: Location of a CSV file containing at least a timestamp column.

    Returns:
        A dataframe with timestamps and placeholder actual values.
    """

    input_path = Path(path)
    if input_path.exists():
        data = pd.read_csv(input_path)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        return data

    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=24, freq="h")
    actual = np.linspace(100.0, 123.0, num=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "actual": actual})


def run_inference(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Run placeholder inference and align predictions with timestamps.

    Args:
        model: Placeholder model object returned by :func:`load_model`.
        data: Input dataframe that should include a timestamp column.

    Returns:
        A dataframe with timestamps and placeholder predictions.
    """

    if "timestamp" not in data.columns:
        timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=len(data), freq="h")
    else:
        timestamps = pd.to_datetime(data["timestamp"])

    base_values = np.arange(len(timestamps), dtype=float)
    predictions = 100.0 + base_values

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "prediction": predictions,
            "model_name": str(model.get("model_name", "unknown_model")),
        }
    )
