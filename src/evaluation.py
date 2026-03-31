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
