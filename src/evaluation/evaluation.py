"""Evaluation utilities for forecast quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _coerce_to_numpy(values: Any) -> np.ndarray:
    """Convert supported inputs into a one-dimensional float array."""

    if isinstance(values, pd.DataFrame):
        if values.empty or values.shape[1] == 0:
            return np.array([], dtype=float)
        values = values.iloc[:, 0]

    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    array = np.asarray(values)
    if array.ndim == 0:
        return np.asarray([array.item()], dtype=float)
    return array.astype(float, copy=False).reshape(-1)


def _align_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce, align, and filter comparable forecast arrays."""

    true_values = _coerce_to_numpy(y_true)
    pred_values = _coerce_to_numpy(y_pred)

    aligned_length = min(len(true_values), len(pred_values))
    if aligned_length == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    true_values = true_values[:aligned_length]
    pred_values = pred_values[:aligned_length]

    valid_mask = np.isfinite(true_values) & np.isfinite(pred_values)
    return true_values[valid_mask], pred_values[valid_mask]


def compute_mae(y_true: Any, y_pred: Any) -> float:
    """Compute mean absolute error from real values."""

    true_values, pred_values = _align_inputs(y_true, y_pred)
    if len(true_values) == 0:
        return 0.0
    return float(np.mean(np.abs(true_values - pred_values)))


def compute_rmse(y_true: Any, y_pred: Any) -> float:
    """Compute root mean squared error from real values."""

    true_values, pred_values = _align_inputs(y_true, y_pred)
    if len(true_values) == 0:
        return 0.0
    return float(np.sqrt(np.mean((true_values - pred_values) ** 2)))


def compute_mape(y_true: Any, y_pred: Any) -> float:
    """Compute mean absolute percentage error from real values."""

    true_values, pred_values = _align_inputs(y_true, y_pred)
    if len(true_values) == 0:
        return 0.0

    non_zero_mask = np.abs(true_values) > 1e-6
    if not np.any(non_zero_mask):
        return 0.0

    true_values = true_values[non_zero_mask]
    pred_values = pred_values[non_zero_mask]
    return float(np.mean(np.abs((true_values - pred_values) / true_values)) * 100.0)


def compute_smape(y_true: Any, y_pred: Any) -> float:
    """Compute symmetric mean absolute percentage error from real values."""

    true_values, pred_values = _align_inputs(y_true, y_pred)
    if len(true_values) == 0:
        return 0.0

    denominator = np.abs(true_values) + np.abs(pred_values)
    valid_mask = denominator > 1e-6
    if not np.any(valid_mask):
        return 0.0

    true_values = true_values[valid_mask]
    pred_values = pred_values[valid_mask]
    denominator = denominator[valid_mask]
    return float(np.mean(2.0 * np.abs(true_values - pred_values) / denominator) * 100.0)


def evaluate_all(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute the standard regression metrics for a forecast."""

    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
    }


def _extract_actual_values(actual: Any) -> np.ndarray:
    """Extract actual values from arrays or project dataframes."""

    if isinstance(actual, pd.DataFrame):
        for column in ("actual", "load_mw", "actual_load_mw", "target"):
            if column in actual.columns:
                return _coerce_to_numpy(actual[column])
        return _coerce_to_numpy(actual)
    return _coerce_to_numpy(actual)


def _extract_predicted_values(predicted: Any) -> np.ndarray:
    """Extract point predictions from arrays or project dataframes."""

    if isinstance(predicted, pd.DataFrame):
        for column in ("prediction", "p50", "forecast", "y_pred", "load_mw"):
            if column in predicted.columns:
                return _coerce_to_numpy(predicted[column])
        return _coerce_to_numpy(predicted)
    return _coerce_to_numpy(predicted)


class ModelEvaluator:
    """Project-friendly forecast evaluator using real metric calculations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the evaluator with project configuration."""

        self.config = config

    def evaluate_forecast(self, actual: Any, predicted: Any) -> dict[str, float]:
        """Evaluate a forecast against actual values.

        Accepts raw arrays or project dataframes and returns the metric naming
        convention expected by the UI and pipeline code.
        """

        y_true = _extract_actual_values(actual)
        y_pred = _extract_predicted_values(predicted)
        summary = evaluate_all(y_true, y_pred)

        return {
            "MAE": summary["mae"],
            "RMSE": summary["rmse"],
            "MAPE": summary["mape"],
            "SMAPE": summary["smape"],
        }
