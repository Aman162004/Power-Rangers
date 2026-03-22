import subprocess
import sys
import os
import yaml
from src.forecast_engine import ForecastEngine
from src.evaluation import ModelEvaluator
from src.forecast_repository import ForecastRepository

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_data_pipeline():
    """Run the data processing pipeline."""
    print("Running data pipeline...")
    subprocess.run([sys.executable, "src/main_pipeline.py"], check=True)

def run_training_pipeline():
    """Run the training pipeline."""
    print("Running training pipeline...")
    subprocess.run([sys.executable, "src/training_pipeline.py"], check=True)

def run_inference_and_evaluation():
    """Run inference and evaluation."""
    print("Running inference and evaluation...")
    config = load_config("config/config.yaml")

    forecast_engine = ForecastEngine("config/config.yaml")
    evaluator = ModelEvaluator(config)
    repository = ForecastRepository(config)

    # Load test data
    test_data = repository.load_dataset("test_data")
    input_data = test_data.tail(24)

    # Generate forecast
    forecast = forecast_engine.generate_forecast(input_data)

    # Evaluate
    actual = test_data.tail(len(forecast))
    metrics = evaluator.evaluate_forecast(actual, forecast)

    print("Evaluation Metrics:")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("Launching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])

def main():
    print("Starting AI Electricity Demand Forecasting System...")

    # Check if data exists
    if not os.path.exists("data/processed/featured_data.parquet"):
        run_data_pipeline()

    # Check if model exists
    if not os.path.exists("models/final_model.pth"):
        run_training_pipeline()

    # Run inference and evaluation
    run_inference_and_evaluation()

    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()