"""Scaffold utilities for identifying forecast peaks."""

from __future__ import annotations

from typing import Any

import pandas as pd


def find_peak(predictions_df: pd.DataFrame) -> dict[str, Any]:
    """Return a placeholder peak value and timestamp from forecast output.

    Args:
        predictions_df: Forecast dataframe containing a timestamp column and a
            prediction-like column.

    Returns:
        A dictionary with `peak_value` and `peak_timestamp`.
    """

    if predictions_df.empty:
        return {"peak_value": None, "peak_timestamp": None}

    value_column = "prediction"
    for candidate in ("prediction", "p50", "load_mw", "actual"):
        if candidate in predictions_df.columns:
            value_column = candidate
            break

    peak_row = predictions_df.loc[predictions_df[value_column].idxmax()]
    timestamp = peak_row["timestamp"] if "timestamp" in peak_row.index else None
    return {
        "peak_value": float(peak_row[value_column]) if pd.notna(peak_row[value_column]) else None,
        "peak_timestamp": str(pd.to_datetime(timestamp)) if timestamp is not None else None,
    }
