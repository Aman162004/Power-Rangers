"""Shared repository helpers for project datasets and model artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch


class ForecastRepository:
    """Small filesystem repository for project data and model artifacts."""

    def __init__(self, config: dict[str, Any]):
        """Initialize paths from the current project configuration."""

        self.config = config
        data_config = config.get("data", {})

        self.feature_engineered_path = data_config.get(
            "historical_feature_engineered_path",
            "data/historical/feature_engineered/",
        )
        self.final_processed_path = data_config.get(
            "historical_final_processed_path",
            "data/historical/final_processed/",
        )
        self.models_path = data_config.get("models_root", "models/")

        os.makedirs(self.feature_engineered_path, exist_ok=True)
        os.makedirs(self.final_processed_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

    def save_dataset(self, df: pd.DataFrame, name: str) -> None:
        """Save a dataset to the expected historical project lane."""

        target_path = self.feature_engineered_path if name == "featured_data" else self.final_processed_path
        path = Path(target_path) / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a dataset from the expected historical project lane."""

        target_path = self.feature_engineered_path if name == "featured_data" else self.final_processed_path
        path = Path(target_path) / f"{name}.parquet"
        return pd.read_parquet(path)

    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        name: str,
    ) -> None:
        """Save a generic PyTorch checkpoint under the configured models root."""

        path = Path(self.models_path) / f"{name}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_model_checkpoint(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Load a generic PyTorch checkpoint and return the saved epoch."""

        path = Path(self.models_path) / f"{name}.pth"
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint["epoch"])
