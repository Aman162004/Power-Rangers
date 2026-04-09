"""
Full system orchestration: data prep -> training -> evaluation -> dashboard.

Pipeline flow:
1. Prepare historical data (if splits don't exist)
2. Train the model (if no active model pointer exists)
3. Run inference & evaluation
4. Launch dashboard
"""

import re
import subprocess
import sys
import os
import json
try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for script mode
    yaml = None
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast.forecast_engine import ForecastEngine
from src.evaluation.evaluation import ModelEvaluator
from src.shared.artifact_repository import ForecastRepository


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    if yaml is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    return {
        "data": {
            "historical_splits_path": "data/historical/final_processed/",
            "training_splits": {
                "train": "train_data.parquet",
                "val": "val_data.parquet",
                "test": "test_data.parquet",
            },
            "models_root": "models/",
            "active_model_pointer": "models/ACTIVE_MODEL.txt",
            "training_drop_columns": [
                "day_of_year",
                "week_of_year",
                "is_weekend",
                "load_mw_raw",
                "load_outlier_detected",
            ],
            "training_drop_pattern": ".*_was_missing$",
        }
    }


def run_data_pipeline():
    """Run the data preparation pipeline."""
    print("[PIPELINE] Running data preparation...")
    subprocess.run([sys.executable, "src/pipelines/prepare_historical_data.py"], check=True)
    print("[PIPELINE] Data preparation complete.")


def run_training_pipeline():
    """Run the training pipeline (consumes prepared splits, outputs checkpoints to models/)."""
    print("[PIPELINE] Running training...")
    subprocess.run([sys.executable, "src/training/training_pipeline.py"], check=True)
    print("[PIPELINE] Training complete.")


def run_inference_and_evaluation():
    """Run inference and evaluation on test set using active model."""
    print("[PIPELINE] Running inference and evaluation...")
    config = load_config("config/config.yaml")
    try:
        repository = ForecastRepository(config)
        forecast_engine = ForecastEngine("config/config.yaml")
        evaluator = ModelEvaluator(config)

        test_data = repository.load_dataset("test_data")

        drop_cols = config['data'].get('training_drop_columns', [])
        drop_cols = [col for col in drop_cols if col in test_data.columns]
        drop_pattern = config['data'].get('training_drop_pattern', '')
        if drop_pattern:
            pattern_cols = [col for col in test_data.columns if re.match(drop_pattern, col)]
            drop_cols.extend(pattern_cols)
        if drop_cols:
            test_data = test_data.drop(columns=list(set(drop_cols)), errors='ignore')

        input_data = test_data.tail(24)
        forecast = forecast_engine.generate_forecast(input_data)
        actual = test_data.tail(len(forecast))
        metrics = evaluator.evaluate_forecast(actual, forecast)
    except Exception as exc:
        print(f"[WARN] Live inference/evaluation unavailable: {exc}")
        saved_metrics_path = PROJECT_ROOT / "models" / "final model" / "forecast_evaluation" / "forecast_metrics.json"
        if not saved_metrics_path.exists():
            raise
        print(f"[PIPELINE] Falling back to saved evaluation bundle: {saved_metrics_path}")
        with saved_metrics_path.open("r", encoding="utf-8") as file:
            saved_metrics = json.load(file)
        metrics = {
            "MAE": float(saved_metrics["MAE"]),
            "RMSE": float(saved_metrics["RMSE"]),
            "MAPE": float(saved_metrics["MAPE"]),
        }

    print("[EVAL] Evaluation Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("[DASHBOARD] Launching dashboard...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/streamlit_frontend/streamlit_app.py"],
            check=True,
        )
    except Exception as exc:
        print(f"[WARN] Dashboard launch skipped: {exc}")


def main():
    print("="*70)
    print("DELHI POWER DEMAND AI - FULL SYSTEM PIPELINE")
    print("="*70)

    config = load_config("config/config.yaml")

    # Step 1: Check if splits exist; prepare data if missing
    print("[CHECK] Checking for prepared training splits...")
    historical_splits_path = Path(config['data']['historical_splits_path'])
    train_split = historical_splits_path / config['data']['training_splits']['train']
    if not train_split.exists():
        print("[CHECK]   -> Splits not found. Running data preparation...")
        run_data_pipeline()
    else:
        print("[CHECK]   -> Splits found. Skipping data preparation.")

    # Step 2: Check if active model exists; train if missing
    print("[CHECK] Checking for trained model...")
    models_root = Path(config['data']['models_root'])
    active_pointer = models_root / config['data']['active_model_pointer'].replace('models/', '')
    final_model_dir = models_root / "final model"
    final_model_ckpts = sorted(final_model_dir.glob("*.ckpt")) if final_model_dir.exists() else []
    if not active_pointer.exists():
        if final_model_ckpts:
            print(f"[CHECK]   -> No active model pointer, but final checkpoint exists: {final_model_ckpts[0].name}")
            print("[CHECK]   -> Skipping training and using the saved final-model checkpoint.")
        else:
            print("[CHECK]   -> No active model. Running training...")
            run_training_pipeline()
    else:
        with open(active_pointer, 'r') as f:
            active_run_id = f.read().strip()
        print(f"[CHECK]   -> Active model found: {active_run_id}")

    # Step 3: Run inference and evaluation
    print("[CHECK] Running inference and evaluation...")
    try:
        run_inference_and_evaluation()
    except Exception as e:
        print(f"[WARN] Inference/evaluation failed: {e}. Continuing to dashboard...")

    # Step 4: Launch dashboard
    print("[CHECK] All systems ready. Launching dashboard...")
    launch_dashboard()


if __name__ == "__main__":
    main()
