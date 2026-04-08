"""Placeholder evaluation helpers for forecast quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_mae(y_true: Any, y_pred: Any) -> float:
    """Return a dummy mean absolute error value.

    Args:
        y_true: Ground-truth values.
        y_pred: Forecasted values.

    Returns:
        A placeholder metric value.
    """

    _ = (np.asarray(y_true), np.asarray(y_pred))
    return 0.0


def compute_rmse(y_true: Any, y_pred: Any) -> float:
    """Return a dummy root mean squared error value.

    Args:
        y_true: Ground-truth values.
        y_pred: Forecasted values.

    Returns:
        A placeholder metric value.
    """

    _ = (np.asarray(y_true), np.asarray(y_pred))
    return 0.0


def compute_mape(y_true: Any, y_pred: Any) -> float:
    """Return a dummy mean absolute percentage error value.

    Args:
        y_true: Ground-truth values.
        y_pred: Forecasted values.

    Returns:
        A placeholder metric value.
    """

    _ = (np.asarray(y_true), np.asarray(y_pred))
    return 0.0


def evaluate_all(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Aggregate placeholder forecast metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Forecasted values.

    Returns:
        A dictionary containing scaffold metric outputs.
    """

    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
    }


class ModelEvaluator:
    """Evaluate forecast quality against actual values."""

    def __init__(self, config: dict):
        """Initialize evaluator with config."""
        self.config = config

    def evaluate_forecast(self, actual: Any, predicted: Any) -> dict[str, float]:
        """
        Evaluate forecast quality.
        
        Placeholder: returns dummy metrics.
        Real implementation will compute MAE, RMSE, MAPE, quantile coverage, etc.
        """
        y_true = np.asarray(actual)
        y_pred = np.asarray(predicted)

        # Placeholder metrics
        mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) > 0 else 0.0
        
        # Avoid division by zero for MAPE
        mape = 0.0
        if len(y_true) > 0 and np.all(y_true != 0):
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
        }
