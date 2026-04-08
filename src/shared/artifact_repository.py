import pandas as pd
import torch
import os
from typing import Tuple

class ForecastRepository:
    def __init__(self, config: dict):
        self.config = config
        feature_engineered_path = config['data'].get('historical_feature_engineered_path', 'data/historical/feature_engineered/')
        final_processed_path = config['data'].get('historical_final_processed_path', 'data/historical/final_processed/')
        models_path = config['data']['models_path']
        os.makedirs(feature_engineered_path, exist_ok=True)
        os.makedirs(final_processed_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)
        self.feature_engineered_path = feature_engineered_path
        self.final_processed_path = final_processed_path
        self.models_path = models_path

    def save_dataset(self, df: pd.DataFrame, name: str):
        """Save DataFrame to the correct historical processing lane."""
        target_path = self.feature_engineered_path if name == "featured_data" else self.final_processed_path
        path = os.path.join(target_path, f"{name}.parquet")
        df.to_parquet(path, index=False)

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load DataFrame from the correct historical processing lane."""
        target_path = self.feature_engineered_path if name == "featured_data" else self.final_processed_path
        path = os.path.join(target_path, f"{name}.parquet")
        return pd.read_parquet(path)

    def save_model_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.models_path, f"{name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_model_checkpoint(self, name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
        """Load model checkpoint and return epoch."""
        path = os.path.join(self.models_path, f"{name}.pth")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']