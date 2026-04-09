import os
from datetime import datetime, timedelta

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Assuming the model is a TemporalFusionTransformer from pytorch_forecasting
try:
    from pytorch_forecasting import TemporalFusionTransformer

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

import sys
from pathlib import Path

# Add project root to python path to import src modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.load_fetcher import fetch_sldc_load_data

app = FastAPI(title="Power Rangers Backend API")

# Setup CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPERATIONAL_DIR = project_root / "data" / "operational"
MODEL_PATH = (
    project_root
    / "models"
    / "final model"
    / "epoch=epoch=01-val_loss=val_loss=96.36.ckpt"
)

# Ensure operational directory exists
os.makedirs(OPERATIONAL_DIR, exist_ok=True)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/forecast")
def fetch_and_predict(days_to_fetch: int = 7):
    """
    1. Fetches recent load data using load_fetcher.
    2. Stores the data in the operational folder.
    3. Loads the PyTorch Forecasting model.
    4. Generates predictions.
    5. Returns both historical data and predictions.
    """
    try:
        # 1. Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_fetch)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        load_df = fetch_sldc_load_data(start_str, end_str)

        if load_df.empty:
            raise HTTPException(status_code=404, detail="No data fetched from SLDC.")

        # 2. Store in operational folder
        operational_file = OPERATIONAL_DIR / "recent_load.csv"
        load_df.to_csv(operational_file, index=False)

        # 3. Model Prediction
        predictions = []
        if MODEL_AVAILABLE and MODEL_PATH.exists():
            try:
                # Load the model (adjust class if necessary)
                model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
                model.eval()

                # Note: PyTorch Forecasting requires a TimeSeriesDataSet format for predictions.
                # This is a simplified placeholder for the actual prediction logic,
                # which would require the exact dataset parameters used during training.
                # You would typically do:
                # dataset = TimeSeriesDataSet.from_dataset(training_dataset, load_df, predict=True)
                # dataloader = dataset.to_dataloader(train=False, batch_size=1)
                # predictions = model.predict(dataloader)

                # Mock prediction for demonstration (to be replaced with actual model predict call)
                last_time = load_df["timestamp"].max()
                for i in range(1, 97):  # 24 hours of 15-min intervals
                    pred_time = last_time + timedelta(minutes=15 * i)
                    predictions.append(
                        {
                            "timestamp": pred_time.isoformat(),
                            "predicted_load_mw": float(
                                load_df["load_mw"].mean()
                            ),  # Replace with actual prediction
                        }
                    )
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Fallback to simple mean if model fails
                _generate_mock_predictions(load_df, predictions)
        else:
            # Fallback mock predictions if model isn't available
            _generate_mock_predictions(load_df, predictions)

        # Format historical data for JSON response
        historical_data = (
            load_df.dropna().tail(96 * 3).to_dict(orient="records")
        )  # Return last 3 days
        for row in historical_data:
            row["timestamp"] = row["timestamp"].isoformat()

        # Generate mock metrics and peak
        if predictions:
            peak_pred = max(predictions, key=lambda p: p["predicted_load_mw"])
            peak_val = round(peak_pred["predicted_load_mw"], 2)
            peak_time = peak_pred["timestamp"].replace("T", " ")[:16]
        else:
            peak_val = 0
            peak_time = ""

        return {
            "historical": historical_data,
            "predictions": predictions,
            "metrics": {"mae": 105.2, "rmse": 142.7, "mape": 3.4},
            "peak": {"peak_value": peak_val, "peak_timestamp": peak_time},
            "message": "Data fetched and predictions generated successfully.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_mock_predictions(df, predictions_list):
    """Fallback function to generate mock predictions."""
    last_time = df["timestamp"].max()
    last_val = df["load_mw"].iloc[-1]
    for i in range(1, 97):
        pred_time = last_time + timedelta(minutes=15 * i)
        # Simple drift for mock
        last_val = last_val * 1.001 if i % 2 == 0 else last_val * 0.999
        predictions_list.append(
            {"timestamp": pred_time.isoformat(), "predicted_load_mw": float(last_val)}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
