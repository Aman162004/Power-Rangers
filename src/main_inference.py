"""Runnable scaffold for placeholder inference, evaluation, and export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for scaffold mode
    yaml = None

from evaluation import evaluate_all
from forecast_engine import load_model, load_test_data, run_inference
from peak_detection import find_peak
from results_exporter import save_metrics, save_peak, save_predictions


def _load_project_config(config_path: Path) -> dict:
    """Load project configuration when available."""

    if yaml is None or not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_checkpoint_path(project_root: Path, config: dict) -> Path:
    """Resolve the preferred model checkpoint path with a safe fallback."""

    candidate_files: list[Path] = []
    final_model_dir = project_root / "models" / "final model"
    runs_dir = project_root / "models" / "runs"

    if final_model_dir.exists():
        candidate_files.extend(sorted(final_model_dir.glob("*.ckpt")))
    if runs_dir.exists():
        candidate_files.extend(sorted(runs_dir.rglob("*.ckpt")))

    if candidate_files:
        return candidate_files[0]

    configured_root = config.get("data", {}).get("models_root")
    if configured_root:
        return project_root / configured_root / "placeholder_model.ckpt"

    return project_root / "models" / "placeholder_model.ckpt"


def _resolve_test_data_path(project_root: Path, config: dict) -> Path:
    """Resolve the preferred test dataset path with a safe fallback."""

    common_candidates = [
        project_root / "data" / "historical" / "final_processed" / "test_data.parquet",
        project_root / "data" / "historical" / "final_processed" / "test_data.csv",
    ]
    for candidate in common_candidates:
        if candidate.exists():
            return candidate

    data_config = config.get("data", {})
    splits_root = data_config.get("historical_splits_path")
    test_file = data_config.get("training_splits", {}).get("test")

    if splits_root and test_file:
        candidate = project_root / splits_root / test_file
        if candidate.exists():
            return candidate

    fallback = project_root / "data" / "test_data.csv"
    return fallback


def _extract_ground_truth(test_data: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
    """Build aligned ground truth values for placeholder evaluation."""

    for column in ("actual", "load_mw", "actual_load_mw"):
        if column in test_data.columns:
            return pd.to_numeric(test_data[column], errors="coerce").fillna(0.0)

    return pd.to_numeric(predictions["prediction"], errors="coerce").fillna(0.0)


def main() -> None:
    """Execute the placeholder inference pipeline end to end."""

    project_root = Path(__file__).resolve().parents[1]
    config = _load_project_config(project_root / "config" / "config.yaml")
    output_dir = project_root / "data" / "outputs"

    checkpoint_path = _resolve_checkpoint_path(project_root, config)
    test_data_path = _resolve_test_data_path(project_root, config)

    model = load_model(str(checkpoint_path))
    test_data = load_test_data(str(test_data_path))
    predictions = run_inference(model, test_data)

    y_true = _extract_ground_truth(test_data, predictions)
    metrics = evaluate_all(y_true=y_true, y_pred=predictions["prediction"])
    peak_info = find_peak(predictions)

    save_predictions(predictions, str(output_dir / "predictions.csv"))
    save_metrics(metrics, str(output_dir / "metrics.json"))
    save_peak(peak_info, str(output_dir / "peak.json"))

    print("Inference scaffold completed successfully.")
    print(f"Checkpoint reference: {checkpoint_path}")
    print(f"Input data reference: {test_data_path}")
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")
    print(f"Peak info saved to: {output_dir / 'peak.json'}")


if __name__ == "__main__":
    main()
