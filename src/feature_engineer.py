import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.

        IMPORTANT: Lag and rolling features depend on the previous N rows of
        `load_mw`. They are only meaningful when the DataFrame is in strict
        chronological order. Per audit §1.1, the upstream SLDC scraper
        concatenates daily pages, which can leave the merged CSV in a
        near-but-not-perfectly-sorted state (especially when one day is
        appended out of order). We therefore sort by timestamp *before* any
        shift/rolling operation, and reset the index so the row positions
        that downstream code (TFT `time_idx`, scaler fits) rely on match
        the chronological order.

        Weather/calendar columns (`temperature`, `humidity`, `wind_speed`,
        `rainfall`, `is_holiday`) are assumed to be already aligned to the
        timestamp in the raw CSV — reordering the rows reorders them too,
        which is correct: each row's weather must travel with its timestamp.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # === Chronological reordering BEFORE any shift/rolling ===
        # Without this, `load_lag_672` would reference whatever row happens
        # to be 672 positions above the current one in the raw CSV, not the
        # row from 672 ticks (7 days) earlier in time. This silently produces
        # nonsense lags that look plausible in spot checks.
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Time-based features
        df = self.create_time_features(df)

        # Cyclical encodings
        df = self.create_cyclical_features(df)

        # Lag features (now safe — df is in chronological order)
        df = self.create_lag_features(df, self.config['features']['lags'])

        # Rolling statistics (now safe — df is in chronological order)
        df = self.create_rolling_features(df, self.config['features']['rolling_windows'])

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        return df

    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encodings for hour."""
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df

    def create_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lag features for load_mw."""
        for lag in lags:
            df[f'load_lag_{lag}'] = df['load_mw'].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling mean features for load_mw."""
        for window in windows:
            df[f'rolling_mean_{window}'] = df['load_mw'].rolling(window=window).mean()
        return df
