"""Fallback data generation when model inference fails."""

from pathlib import Path
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dummy_forecast_data() -> dict:
    """
    Load dummy forecast data from historical training data.
    
    When inference fails and ENABLE_DUMMY_FALLBACK is true, this returns
    a realistic dummy forecast sampled from historical data.
    
    Returns:
        dict: Response structure matching the real /api/forecast endpoint
    """
    try:
        # Load training data
        train_path = PROJECT_ROOT / "data" / "historical" / "final_processed" / "train_data.parquet"
        
        if not train_path.exists():
            logger.warning(f"Training data not found at {train_path}. Using minimal fallback.")
            return _generate_minimal_fallback()
        
        train_df = pd.read_parquet(train_path)
        
        if train_df.empty:
            logger.warning("Training data is empty. Using minimal fallback.")
            return _generate_minimal_fallback()
        
        # Ensure timestamp column exists
        if "timestamp" not in train_df.columns:
            logger.warning("Timestamp column not found in training data. Using minimal fallback.")
            return _generate_minimal_fallback()
        
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
        train_df = train_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        
        if train_df.empty:
            logger.warning("No valid timestamps in training data. Using minimal fallback.")
            return _generate_minimal_fallback()
        
        # Sample a random contiguous 48-hour period from training data (96 15-min intervals)
        # to create realistic predictions
        available_indices = range(len(train_df) - 96)
        if available_indices:
            start_idx = np.random.choice(available_indices)
            sample_df = train_df.iloc[start_idx : start_idx + 96].copy()
        else:
            sample_df = train_df.tail(96).copy()
        
        if sample_df.empty:
            logger.warning("Could not sample data. Using minimal fallback.")
            return _generate_minimal_fallback()
        
        # Build predictions list with realistic values from sampled data
        load_col = "load_mw" if "load_mw" in sample_df.columns else None
        if not load_col:
            # Try to find the load column by name pattern
            load_cols = [col for col in sample_df.columns if "load" in col.lower() and "mw" in col.lower()]
            if load_cols:
                load_col = load_cols[0]
        
        predictions = []
        historical_data = []
        
        if load_col and load_col in sample_df.columns:
            # Use actual values with realistic variance
            base_values = sample_df[load_col].values
            
            for idx, row in sample_df.iterrows():
                timestamp = row["timestamp"]
                base_load = float(row[load_col]) if pd.notna(row[load_col]) else 40000.0
                
                # Add realistic variance: ±5% from base with p10/p50/p90
                noise = np.random.normal(0, base_load * 0.02)  # 2% std dev
                p50_val = base_load + noise
                p10_val = p50_val * 0.95  # 5% below p50
                p90_val = p50_val * 1.05  # 5% above p50
                
                # Clamp to reasonable ranges (20k-60k MW is typical)
                p50_val = np.clip(p50_val, 20000, 60000)
                p10_val = np.clip(p10_val, 20000, 60000)
                p90_val = np.clip(p90_val, 20000, 60000)
                
                predictions.append({
                    "timestamp": timestamp.isoformat(),
                    "predicted_load_mw": float(p50_val),
                    "p10": float(p10_val),
                    "p50": float(p50_val),
                    "p90": float(p90_val),
                    "actual_load_mw": None,  # No actual values for forecasts
                })
            
            # Last 3 days of historical data (same sample as we're using for forecast)
            for idx, row in sample_df.iterrows():
                hist_row = {
                    "timestamp": row["timestamp"].isoformat(),
                    "load_mw": float(row[load_col]) if pd.notna(row[load_col]) else None,
                }
                # Add any other columns that might be useful
                for col in sample_df.columns:
                    if col not in ["timestamp", load_col]:
                        hist_row[col] = row[col]
                historical_data.append(hist_row)
        else:
            # Fallback with generic values
            return _generate_minimal_fallback()
        
        # Compute basic metrics
        pred_values = [p["p50"] for p in predictions]
        mae = float(np.mean([abs(v - np.mean(pred_values)) for v in pred_values]))
        rmse = float(np.sqrt(np.mean([(v - np.mean(pred_values))**2 for v in pred_values])))
        mape = 2.5  # Dummy MAPE
        
        # Peak prediction
        if predictions:
            peak_pred = max(predictions, key=lambda p: p["predicted_load_mw"])
            peak_val = round(peak_pred["predicted_load_mw"], 2)
            peak_time = peak_pred["timestamp"].replace("T", " ")[:16]
        else:
            peak_val = 0
            peak_time = ""
        
        # Use today's date for dummy forecast
        now = datetime.now()
        forecast_date = now.strftime("%Y-%m-%d")
        history_end = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        return {
            "historical": historical_data,
            "predictions": predictions,
            "metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
            },
            "peak": {
                "peak_value": peak_val,
                "peak_timestamp": peak_time,
            },
            "forecast_date": forecast_date,
            "history_end_date": history_end,
            "aggressiveness_pct": 0.0,
            "temperature_delta_c": None,
            "message": "Dummy data (inference failed)",
        }
    except Exception as e:
        logger.exception(f"Error generating dummy forecast data: {e}")
        return _generate_minimal_fallback()


def _generate_minimal_fallback() -> dict:
    """Generate a minimal fallback response with generic constant values."""
    now = datetime.now()
    predictions = []
    
    # Generate 48 hours of predictions with constant ~40000 MW
    for i in range(96):  # 96 * 15min = 24 hours (adjust to 192 for 48 hours if needed)
        timestamp = now + timedelta(minutes=i * 15)
        base_val = 40000.0
        predictions.append({
            "timestamp": timestamp.isoformat(),
            "predicted_load_mw": base_val,
            "p10": base_val * 0.95,
            "p50": base_val,
            "p90": base_val * 1.05,
            "actual_load_mw": None,
        })
    
    # Minimal historical data
    historical_data = []
    for i in range(96):
        timestamp = now - timedelta(days=1) + timedelta(minutes=i * 15)
        historical_data.append({
            "timestamp": timestamp.isoformat(),
            "load_mw": 40000.0,
        })
    
    return {
        "historical": historical_data,
        "predictions": predictions,
        "metrics": {
            "mae": 500.0,
            "rmse": 650.0,
            "mape": 1.25,
        },
        "peak": {
            "peak_value": 42000.0,
            "peak_timestamp": (now + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
        },
        "forecast_date": now.strftime("%Y-%m-%d"),
        "history_end_date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "aggressiveness_pct": 0.0,
        "temperature_delta_c": None,
        "message": "Dummy data (inference failed)",
    }
