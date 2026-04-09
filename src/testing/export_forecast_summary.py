"""Export compact forecast artifacts from the saved final-model evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTING_ROOT = PROJECT_ROOT / "models" / "testing"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final model"
FINAL_CHECKPOINT = FINAL_MODEL_DIR / "epoch=epoch=01-val_loss=val_loss=96.36.ckpt"
OUTPUT_DIR = FINAL_MODEL_DIR / "forecast_evaluation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation import evaluate_all


def _latest_testing_run() -> Path:
    """Return the most recently updated testing run folder."""

    run_dirs = [path for path in TESTING_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No testing runs found under {TESTING_ROOT}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _load_metrics(metrics_path: Path) -> dict:
    """Load persisted metrics, tolerating NaN values in JSON."""

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _build_peak_info(predictions_df: pd.DataFrame) -> dict[str, object]:
    """Find the highest forecasted median demand and its timestamp."""

    peak_row = predictions_df.loc[predictions_df["p50"].idxmax()]
    return {
        "peak_timestamp": str(pd.to_datetime(peak_row["timestamp"])),
        "peak_value_mw": float(peak_row["p50"]),
        "actual_at_peak_mw": float(peak_row["actual_load_mw"]),
    }


def _build_compact_results(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact subset of the forecast output columns."""

    compact = predictions_df.loc[:, ["timestamp", "actual_load_mw", "p10", "p50", "p90"]].copy()
    compact["forecast_peak_flag"] = compact["p50"] == compact["p50"].max()
    return compact


def _build_project_validation_summary(
    source_run: Path,
    predictions_df: pd.DataFrame,
    recalculated_metrics: dict[str, float],
) -> str:
    """Create a concise validation summary for the available project checks."""

    test_data_path = PROJECT_ROOT / "data" / "historical" / "final_processed" / "test_data.parquet"
    checks = [
        f"Checkpoint exists: {'PASS' if FINAL_CHECKPOINT.exists() else 'FAIL'}",
        f"Source evaluation run exists: {'PASS' if source_run.exists() else 'FAIL'}",
        f"Source prediction file exists: {'PASS' if (source_run / 'test_predictions_vs_actual.csv').exists() else 'FAIL'}",
        f"Historical test data exists: {'PASS' if test_data_path.exists() else 'FAIL'}",
        f"Evaluation row count > 0: {'PASS' if len(predictions_df) > 0 else 'FAIL'}",
        f"Peak detected: {'PASS' if predictions_df['p50'].notna().any() else 'FAIL'}",
        "Direct checkpoint replay in current shell: BLOCKED (torch is not installed in this Python environment)",
    ]

    summary_lines = [
        "Project Validation Summary",
        f"Final checkpoint: {FINAL_CHECKPOINT}",
        f"Evaluation source run: {source_run}",
        f"Rows evaluated: {len(predictions_df)}",
        f"MAE: {recalculated_metrics['mae']:.4f}",
        f"RMSE: {recalculated_metrics['rmse']:.4f}",
        f"MAPE: {recalculated_metrics['mape']:.4f}%",
        "",
        "Checks:",
        *checks,
    ]
    return "\n".join(summary_lines) + "\n"


def main() -> None:
    """Save compact forecast results and summaries in one final-model folder."""

    source_run = _latest_testing_run()
    predictions_path = source_run / "test_predictions_vs_actual.csv"
    metrics_path = source_run / "metrics.json"

    predictions_df = pd.read_csv(predictions_path)
    predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"])

    recalculated_metrics = evaluate_all(
        predictions_df["actual_load_mw"],
        predictions_df["p50"],
    )
    saved_metrics = _load_metrics(metrics_path)
    peak_info = _build_peak_info(predictions_df)
    compact_results = _build_compact_results(predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compact_csv_path = OUTPUT_DIR / "forecast_results_small.csv"
    compact_json_path = OUTPUT_DIR / "forecast_results_small.json"
    peak_json_path = OUTPUT_DIR / "forecast_peak.json"
    metrics_json_path = OUTPUT_DIR / "forecast_metrics.json"
    summary_path = OUTPUT_DIR / "forecast_metrics_summary.txt"
    validation_path = OUTPUT_DIR / "project_validation_summary.txt"

    compact_results.to_csv(compact_csv_path, index=False)
    compact_results.to_json(compact_json_path, orient="records", indent=2, date_format="iso")

    enriched_metrics = {
        "checkpoint_path": str(FINAL_CHECKPOINT),
        "source_evaluation_run": str(source_run),
        "rows_evaluated": int(len(predictions_df)),
        "MAE": recalculated_metrics["mae"],
        "RMSE": recalculated_metrics["rmse"],
        "MAPE": recalculated_metrics["mape"],
        "SMAPE": recalculated_metrics["smape"],
        "saved_metrics_snapshot": saved_metrics,
    }

    peak_json_path.write_text(json.dumps(peak_info, indent=2), encoding="utf-8")
    metrics_json_path.write_text(json.dumps(enriched_metrics, indent=2), encoding="utf-8")

    summary_lines = [
        f"Checkpoint: {FINAL_CHECKPOINT}",
        f"Source evaluation run: {source_run}",
        f"Rows evaluated: {len(predictions_df)}",
        f"MAE: {recalculated_metrics['mae']:.4f}",
        f"RMSE: {recalculated_metrics['rmse']:.4f}",
        f"MAPE: {recalculated_metrics['mape']:.4f}%",
        f"SMAPE: {recalculated_metrics['smape']:.4f}%",
        f"Peak forecast: {peak_info['peak_value_mw']:.4f} MW",
        f"Peak time: {peak_info['peak_timestamp']}",
        f"Actual at peak time: {peak_info['actual_at_peak_mw']:.4f} MW",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    validation_path.write_text(
        _build_project_validation_summary(source_run, predictions_df, recalculated_metrics),
        encoding="utf-8",
    )

    print(f"Final checkpoint: {FINAL_CHECKPOINT}")
    print(f"Source run: {source_run}")
    print(f"Saved compact CSV: {compact_csv_path}")
    print(f"Saved compact JSON: {compact_json_path}")
    print(f"Saved peak JSON: {peak_json_path}")
    print(f"Saved metrics JSON: {metrics_json_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved validation summary: {validation_path}")
    print("Metrics:")
    print(f"  MAE={recalculated_metrics['mae']:.4f}")
    print(f"  RMSE={recalculated_metrics['rmse']:.4f}")
    print(f"  MAPE={recalculated_metrics['mape']:.4f}%")


if __name__ == "__main__":
    main()
