import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, encoder_window: int, decoder_window: int, stride: int = 1):
        self.encoder_window = encoder_window
        self.decoder_window = decoder_window
        self.stride = stride

        # Assume target is load_mw, features are all except timestamp and load_mw
        self.target_col = 'load_mw'
        self.feature_cols = [col for col in data.columns if col not in ['timestamp', self.target_col]]

        # Convert to numpy
        self.features = data[self.feature_cols].values.astype(np.float32)
        self.targets = data[self.target_col].values.astype(np.float32)

        self.length = len(data) - encoder_window - decoder_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        encoder_end = start_idx + self.encoder_window
        decoder_end = encoder_end + self.decoder_window

        # Encoder input: features for encoder_window
        encoder_input = self.features[start_idx:encoder_end]
        # Decoder target: load_mw for decoder_window
        decoder_target = self.targets[encoder_end:decoder_end]

        return torch.tensor(encoder_input), torch.tensor(decoder_target)


class DatasetBuilder:
    def __init__(self, config: dict, feature_engineer):
        self.config = config
        self.feature_engineer = feature_engineer
        self.last_preprocess_report = {}

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw CSV data."""
        return pd.read_csv(file_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with timestamp normalization and transparent missing handling."""
        if 'timestamp' not in df.columns:
            raise ValueError("Input data must contain a 'timestamp' column.")

        work_df = df.copy()
        work_df['timestamp'] = pd.to_datetime(work_df['timestamp'], errors='coerce')
        invalid_timestamp_rows = int(work_df['timestamp'].isna().sum())
        work_df = work_df.dropna(subset=['timestamp'])

        pre_dedup_rows = len(work_df)
        work_df = work_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
        duplicate_rows_removed = int(pre_dedup_rows - len(work_df))

        value_columns = [col for col in work_df.columns if col != 'timestamp']
        missing_counts_before_fill = work_df[value_columns].isna().sum().to_dict()

        # Keep explicit missing indicators so downstream users can inspect data quality.
        for col in value_columns:
            work_df[f'{col}_was_missing'] = work_df[col].isna().astype(int)

        work_df[value_columns] = work_df[value_columns].ffill().bfill()

        # Final safety net: fill any residual values using simple per-column statistics.
        for col in value_columns:
            if work_df[col].isna().any():
                if np.issubdtype(work_df[col].dtype, np.number):
                    work_df[col] = work_df[col].fillna(work_df[col].median())
                else:
                    mode_values = work_df[col].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
                    work_df[col] = work_df[col].fillna(fill_value)

        self.last_preprocess_report = {
            'rows_input': int(len(df)),
            'rows_after_timestamp_cleanup': int(len(work_df)),
            'invalid_timestamp_rows': invalid_timestamp_rows,
            'duplicate_rows_removed': duplicate_rows_removed,
            'missing_counts_before_fill': missing_counts_before_fill,
        }

        return work_df

    def split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe chronologically into train/val/test with 70/15/15 ratio."""
        n = len(df)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)
        return train_df, val_df, test_df

    def build_datasets(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """Build train, val, test datasets using pytorch_forecasting."""
        lag_cols = [f'load_lag_{lag}' for lag in self.config['features']['lags']]
        rolling_cols = [f'rolling_mean_{w}' for w in self.config['features']['rolling_windows']]
        required_feature_cols = ['hour', 'day_of_week', 'month'] + lag_cols + rolling_cols

        if all(col in df.columns for col in required_feature_cols):
            df_featured = df.copy().reset_index(drop=True)
        else:
            df_featured = self.feature_engineer.engineer_features(df)
            df_featured = df_featured.dropna().reset_index(drop=True)

        # Add time_idx and group_id for TFT
        if 'time_idx' not in df_featured.columns:
            df_featured['time_idx'] = range(len(df_featured))
        if 'group_id' not in df_featured.columns:
            df_featured['group_id'] = 0  # Single series

        # Define features
        static_categoricals = []
        static_reals = []
        time_varying_known_categoricals = ['hour', 'day_of_week', 'month']
        time_varying_known_reals = ['sin_hour', 'cos_hour', 'temperature', 'humidity', 'wind_speed', 'rainfall']
        time_varying_unknown_categoricals = []
        time_varying_unknown_reals = ['load_mw'] + [f'load_lag_{lag}' for lag in self.config['features']['lags']] + [f'rolling_mean_{w}' for w in self.config['features']['rolling_windows']]

        # Split
        train_df, val_df, test_df = self.split_dataframe(df_featured)

        max_encoder_length = self.config['pipeline']['encoder_window']
        max_prediction_length = self.config['pipeline']['decoder_window']

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="load_mw",
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
        test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
        return training, val_dataset, test_dataset
