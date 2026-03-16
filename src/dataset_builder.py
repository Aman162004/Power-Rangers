import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

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

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw CSV data."""
        return pd.read_csv(file_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing: sort by timestamp, handle missing values."""
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Simple forward fill for missing values
        df = df.fillna(method='ffill')
        return df

    def build_datasets(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """Build train, val, test datasets."""
        # Apply feature engineering
        df_featured = self.feature_engineer.engineer_features(df)
        # Drop NaN from lags and rolling
        df_featured = df_featured.dropna().reset_index(drop=True)

        # Split into train/val/test (e.g., 70/15/15)
        n = len(df_featured)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_df = df_featured[:train_end]
        val_df = df_featured[train_end:val_end]
        test_df = df_featured[val_end:]

        # Create datasets
        train_ds = TimeSeriesDataset(train_df, self.config['pipeline']['encoder_window'],
                                     self.config['pipeline']['decoder_window'], self.config['pipeline']['stride'])
        val_ds = TimeSeriesDataset(val_df, self.config['pipeline']['encoder_window'],
                                   self.config['pipeline']['decoder_window'], self.config['pipeline']['stride'])
        test_ds = TimeSeriesDataset(test_df, self.config['pipeline']['encoder_window'],
                                    self.config['pipeline']['decoder_window'], self.config['pipeline']['stride'])

        return train_ds, val_ds, test_ds