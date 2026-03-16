import pandas as pd
import torch
import os
from typing import Tuple

class ForecastRepository:
    def __init__(self, config: dict):
        self.config = config
        os.makedirs(self.config['data']['processed_path'], exist_ok=True)
        os.makedirs(self.config['data']['models_path'], exist_ok=True)

    def save_dataset(self, df: pd.DataFrame, name: str):
        """Save DataFrame to Parquet."""
        path = os.path.join(self.config['data']['processed_path'], f"{name}.parquet")
        df.to_parquet(path, index=False)

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load DataFrame from Parquet."""
        path = os.path.join(self.config['data']['processed_path'], f"{name}.parquet")
        return pd.read_parquet(path)

    def save_model_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.config['data']['models_path'], f"{name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_model_checkpoint(self, name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
        """Load model checkpoint and return epoch."""
        path = os.path.join(self.config['data']['models_path'], f"{name}.pth")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']