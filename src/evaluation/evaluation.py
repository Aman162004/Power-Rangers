"""Evaluation utilities for forecast quality checks."""

from __future__ import annotations

import warnings
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


def segmented_mape(
    backtest_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    min_samples: int = 15,
) -> dict[str, Any]:
    """Compute per-segment MAPE from a walk-forward backtest DataFrame.

    Segments (Delhi-specific seasonality, per the audit prompt):
      - `by_season`: summer (Apr-Jun), monsoon (Jul-Sep), winter (Dec-Feb),
        shoulder (Mar, Oct, Nov). Driven by the month of the predicted timestamp.
      - `extreme_heat_gt40C`: predicted timestamps whose recorded temperature
        exceeds 40 C (requires `full_df` with a `temperature` column).
      - `peak_stress_top5pct`: predicted timestamps that fall in the top 5% of
        the recorded load distribution (requires `full_df` with a `load_mw`
        column). This is the 95th-percentile stress test.

    Sample-size discipline (audit round 2): any segment with fewer than
    `min_samples` observations raises a `UserWarning` so the reported mean/std
    is treated as directional only, never as a statistically reliable estimate.

    Returns a dict with keys `by_season` (DataFrame), `extreme_heat_gt40C`
    (Series), `peak_stress_top5pct` (Series), and `min_samples` (int).
    """
    if backtest_df is None or len(backtest_df) == 0:
        raise ValueError("segmented_mape: backtest_df is empty (run walk_forward_backtest first).")

    df = backtest_df.copy()
    df["origin"] = pd.to_datetime(df["origin"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "mape" not in df.columns and "ape_p50" in df.columns:
        df["mape"] = df["ape_p50"]

    out: dict[str, Any] = {}

    # 1. Season segmentation (Delhi climatology).
    df["month"] = df["timestamp"].dt.month
    season_map = {
        **{m: "summer" for m in [4, 5, 6]},
        **{m: "monsoon" for m in [7, 8, 9]},
        **{m: "winter" for m in [12, 1, 2]},
    }
    df["season"] = df["month"].map(season_map).fillna("shoulder")
    season_order = ["summer", "monsoon", "winter", "shoulder"]
    by_season = df.groupby("season")["mape"].agg(["mean", "std", "count"])
    by_season = by_season.reindex([s for s in season_order if s in by_season.index])

    # 2. Extreme-heat segmentation (temperature > 40 C).
    extreme_heat = pd.Series(dtype=float, name="mape")
    if full_df is not None and "temperature" in full_df.columns:
        temp_map = full_df[["timestamp", "temperature"]].drop_duplicates("timestamp")
        temp_map["timestamp"] = pd.to_datetime(temp_map["timestamp"], errors="coerce")
        df = df.merge(temp_map, on="timestamp", how="left")
        heat = df[df["temperature"] > 40]
        if not heat.empty:
            extreme_heat = heat["mape"].agg(["mean", "std", "count"])
    else:
        warnings.warn(
            "segmented_mape: full_df with a `temperature` column was not supplied — "
            "the extreme_heat_gt40C segment will be empty.",
            UserWarning,
            stacklevel=2,
        )

    # 3. Peak-stress segmentation (top 5% of recorded load).
    peak_stress = pd.Series(dtype=float, name="mape")
    if full_df is not None and "load_mw" in full_df.columns:
        load_threshold = float(full_df["load_mw"].quantile(0.95))
        peak_ts = set(full_df.loc[full_df["load_mw"] > load_threshold, "timestamp"])
        peak_ts = {pd.Timestamp(t) for t in peak_ts}
        stress = df[df["timestamp"].isin(peak_ts)]
        if not stress.empty:
            peak_stress = stress["mape"].agg(["mean", "std", "count"])
    else:
        warnings.warn(
            "segmented_mape: full_df with a `load_mw` column was not supplied — "
            "the peak_stress_top5pct segment will be empty.",
            UserWarning,
            stacklevel=2,
        )

    # 4. Sample-size discipline: warn loudly when a segment is too small.
    def _warn_on_small(segment_label: str, stats: Any) -> None:
        count = stats.get("count", 0) if isinstance(stats, dict) else 0
        if count > 0 and count < min_samples:
            warnings.warn(
                f"segmented_mape: segment '{segment_label}' has only {count} samples "
                f"(< min_samples={min_samples}). Treat its mean/std as directional "
                "only — not a statistically reliable estimate.",
                UserWarning,
                stacklevel=3,
            )

    for season_name, season_row in by_season.iterrows():
        _warn_on_small(f"by_season/{season_name}", season_row.to_dict())
    _warn_on_small("extreme_heat_gt40C", extreme_heat.to_dict() if not extreme_heat.empty else {})
    _warn_on_small("peak_stress_top5pct", peak_stress.to_dict() if not peak_stress.empty else {})

    out["by_season"] = by_season
    out["extreme_heat_gt40C"] = extreme_heat
    out["peak_stress_top5pct"] = peak_stress
    out["min_samples"] = min_samples
    return out


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
