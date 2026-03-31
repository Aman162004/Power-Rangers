"""Runnable scaffold for the AI forecasting inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation import evaluate_all
from forecast_engine import load_model, load_test_data, run_inference
from peak_detection import find_peak
from results_exporter import save_metrics, save_peak, save_predictions


def build_dummy_ground_truth(predictions_df: pd.DataFrame) -> pd.Series:
    """Create dummy ground-truth values aligned with placeholder predictions."""

    return predictions_df["prediction"] - 1.5


def main() -> None:
    """Execute the placeholder inference, evaluation, and export pipeline."""

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "outputs"

    model = load_model(str(project_root / "artifacts" / "placeholder_model.ckpt"))
    test_data = load_test_data(str(project_root / "data" / "test_data.csv"))
    predictions = run_inference(model, test_data)

    y_true = (
        test_data["actual"]
        if "actual" in test_data.columns
        else build_dummy_ground_truth(predictions)
    )
    metrics = evaluate_all(y_true=y_true, y_pred=predictions["prediction"])
    peak_info = find_peak(predictions)

    save_predictions(predictions, str(output_dir / "predictions.csv"))
    save_metrics(metrics, str(output_dir / "metrics.json"))
    save_peak(peak_info, str(output_dir / "peak.json"))

    print("Inference scaffold completed successfully.")
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")
    print(f"Peak info saved to: {output_dir / 'peak.json'}")


if __name__ == "__main__":
    main()
