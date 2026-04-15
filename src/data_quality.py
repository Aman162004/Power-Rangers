from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


class DataQualityReporter:
    """Generate a small JSON-serializable quality summary for preprocessing outputs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _stage_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        null_counts = {column: int(count) for column, count in df.isna().sum().to_dict().items()}
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "null_counts": null_counts,
            "numeric_columns": numeric_columns,
        }

    def generate_full_report(
        self,
        raw_df: pd.DataFrame,
        featured_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build a compact report used by the historical prep pipeline."""
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stages": {
                "raw": self._stage_summary(raw_df),
                "featured": self._stage_summary(featured_df),
                "processed": self._stage_summary(cleaned_df),
            },
            "row_flow": {
                "raw_rows": int(len(raw_df)),
                "featured_rows": int(len(featured_df)),
                "processed_rows": int(len(cleaned_df)),
                "dropped_raw_to_featured": int(len(raw_df) - len(featured_df)),
                "dropped_featured_to_processed": int(len(featured_df) - len(cleaned_df)),
            },
        }
