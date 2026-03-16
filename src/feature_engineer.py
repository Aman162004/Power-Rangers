import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df = self.create_time_features(df)

        # Cyclical encodings
        df = self.create_cyclical_features(df)

        # Lag features
        df = self.create_lag_features(df, self.config['features']['lags'])

        # Rolling statistics
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