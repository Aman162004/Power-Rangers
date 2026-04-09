"""Basic export helpers for scaffold prediction outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_predictions(df: pd.DataFrame, path: str) -> None:
    """Save predictions to CSV.

    Args:
        df: Prediction dataframe to persist.
        path: Output CSV path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_metrics(metrics: dict[str, Any], path: str) -> None:
    """Save metric data to JSON.

    Args:
        metrics: Metric dictionary to persist.
        path: Output JSON path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)


def save_peak(peak_info: dict[str, Any], path: str) -> None:
    """Save peak metadata to JSON.

    Args:
        peak_info: Peak summary dictionary to persist.
        path: Output JSON path.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(peak_info, file, indent=2, default=str)
