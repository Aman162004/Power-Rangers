"""Placeholder utilities for detecting forecast peaks."""

from __future__ import annotations

from typing import Any

import pandas as pd


def find_peak(predictions_df: pd.DataFrame) -> dict[str, Any]:
    """Return the peak prediction value and its timestamp.

    The current implementation uses a simple maximum over the `prediction`
    column so integration work can proceed before the real peak-detection logic
    is designed.

    Args:
        predictions_df: Dataframe containing `timestamp` and `prediction`.

    Returns:
        A dictionary with `peak_value` and `peak_timestamp`.
    """

    if predictions_df.empty:
        return {"peak_value": None, "peak_timestamp": None}

    peak_row = predictions_df.loc[predictions_df["prediction"].idxmax()]
    return {
        "peak_value": float(peak_row["prediction"]),
        "peak_timestamp": str(pd.to_datetime(peak_row["timestamp"])),
    }
