"""Data ingestion package for external load and weather sources."""

from src.ingestion.pipeline import run_ingestion

__all__ = ["run_ingestion"]
