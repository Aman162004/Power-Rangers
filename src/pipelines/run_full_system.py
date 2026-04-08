"""
Full system orchestration: data prep → training → evaluation → dashboard.

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
import yaml
from pathlib import Path

from src.forecast_engine import ForecastEngine
from src.evaluation import ModelEvaluator
from src.shared.artifact_repository import ForecastRepository


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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

    repository = ForecastRepository(config)
    forecast_engine = ForecastEngine("config/config.yaml")
    evaluator = ModelEvaluator(config)

    # Load test data
    test_data = repository.load_dataset("test_data")
    
    # Remove audit columns (same as training does)
    drop_cols = config['data'].get('training_drop_columns', [])
    drop_cols = [col for col in drop_cols if col in test_data.columns]
    drop_pattern = config['data'].get('training_drop_pattern', '')
    if drop_pattern:
        pattern_cols = [col for col in test_data.columns if re.match(drop_pattern, col)]
        drop_cols.extend(pattern_cols)
    if drop_cols:
        test_data = test_data.drop(columns=list(set(drop_cols)), errors='ignore')
    
    input_data = test_data.tail(24)

    # Generate forecast
    forecast = forecast_engine.generate_forecast(input_data)

    # Evaluate
    actual = test_data.tail(len(forecast))
    metrics = evaluator.evaluate_forecast(actual, forecast)

    print("[EVAL] Evaluation Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("[DASHBOARD] Launching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_frontend/streamlit_app.py"])


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
        print("[CHECK]   → Splits not found. Running data preparation...")
        run_data_pipeline()
    else:
        print("[CHECK]   → Splits found. Skipping data preparation.")

    # Step 2: Check if active model exists; train if missing
    print("[CHECK] Checking for trained model...")
    models_root = Path(config['data']['models_root'])
    active_pointer = models_root / config['data']['active_model_pointer'].replace('models/', '')
    if not active_pointer.exists():
        print("[CHECK]   → No active model. Running training...")
        run_training_pipeline()
    else:
        with open(active_pointer, 'r') as f:
            active_run_id = f.read().strip()
        print(f"[CHECK]   → Active model found: {active_run_id}")

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
