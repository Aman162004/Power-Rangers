"""Fallback data generation when model inference fails."""

from pathlib import Path
from datetime import datetime, timedelta
import logging

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dummy_forecast_data() -> dict:
    """
    Generate fallback forecast data using historical data from 10 days ago.
    
    When model inference fails and ENABLE_DUMMY_FALLBACK is true:
    - Actual historical data is shown as-is
    - Predicted data uses load values from 10 days ago (p50)
    - p10 and p90 are calculated as ±5% from p50
    
    Returns:
        dict: Response structure matching /api/forecast response
    """
    try:
        now = datetime.now()
        
        # Try to load recent historical data
        operational_file = PROJECT_ROOT / "data" / "operational" / "recent_load.csv"
        
        if operational_file.exists():
            try:
                df = pd.read_csv(operational_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                
                if not df.empty:
                    return _generate_fallback_from_history(df, now)
            except Exception as e:
                logger.warning(f"Failed to load operational data: {e}")
        
        # Fallback: try to load from historical processed data
        train_path = PROJECT_ROOT / "data" / "historical" / "final_processed" / "train_data.parquet"
        if train_path.exists():
            try:
                df = pd.read_parquet(train_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                
                if not df.empty:
                    return _generate_fallback_from_history(df, now)
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")
        
        # Last resort: simple fallback
        logger.warning("Could not load any historical data. Using minimal fallback.")
        return _generate_minimal_fallback()
        
    except Exception as e:
        logger.exception(f"Error generating fallback forecast data: {e}")
        return _generate_minimal_fallback()


def _generate_fallback_from_history(df: pd.DataFrame, now: datetime) -> dict:
    """
    Generate fallback forecast using historical data.
    
    - Actual: Show all available data (both historical and any fetched for forecast window)
    - Predicted: Use data sampled from earlier in the dataset, shifted to forecast window
    
    Args:
        df: Historical dataframe with timestamp and load_mw columns (from recent_load.csv or train_data)
        now: Current datetime
    
    Returns:
        dict: Fallback response with actual and predicted data
    """
    try:
        # Ensure we have the right columns
        load_col = "load_mw" if "load_mw" in df.columns else None
        if not load_col:
            load_cols = [col for col in df.columns if "load" in col.lower() and "mw" in col.lower()]
            if load_cols:
                load_col = load_cols[0]
        
        if not load_col:
            logger.warning("Could not find load column in data")
            return _generate_minimal_fallback()
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Separate historical data (past) and actual data (forecast window)
        # Historical = last 3 days of actual data
        historical_data = []
        history_df = df.tail(288).copy() if len(df) > 288 else df.copy()
        
        for idx, row in history_df.iterrows():
            historical_data.append({
                "timestamp": row["timestamp"].isoformat(),
                "load_mw": float(row[load_col]) if pd.notna(row[load_col]) else None,
            })
        
        # Get actual data that's already in the dataset (most recent data is forecasted period)
        all_timestamps = set(df["timestamp"].dt.date.unique())
        max_date = df["timestamp"].max().date()
        min_date = df["timestamp"].min().date()
        
        # Data from last 1 day might be actual/near-real forecast data
        last_day_threshold = df["timestamp"].max() - timedelta(days=1)
        actual_data_df = df[df["timestamp"] >= last_day_threshold].copy()
        
        # Build actual load dict for quick lookup
        actual_loads = {}
        for idx, row in actual_data_df.iterrows():
            ts_key = row["timestamp"]
            actual_loads[ts_key] = float(row[load_col]) if pd.notna(row[load_col]) else None
        
        # For predictions: use data from earlier in dataset as a pattern
        prediction_source = []
        
        if len(df) > 1440:
            # Use data from ~10 days earlier
            start_idx = max(0, len(df) - 1440 - 96)
            prediction_source = df.iloc[start_idx : start_idx + 96].copy()
        elif len(df) > 288:
            # Use data from earlier in the dataset
            start_idx = max(0, len(df) - 577)
            prediction_source = df.iloc[start_idx : start_idx + 96].copy()
        else:
            # Not enough data, use what we have
            prediction_source = df.tail(96).copy()
        
        # Build predictions from the source data
        predictions = []
        
        if not prediction_source.empty:
            # Get the timestamp range of prediction_source
            source_start = prediction_source["timestamp"].min()
            
            # Calculate how many days the source is from max date
            days_offset = (max_date - source_start.date()).days
            if days_offset < 1:
                days_offset = 10
            
            for idx, row in prediction_source.iterrows():
                old_timestamp = row["timestamp"]
                # Shift this historical data forward to forecast window
                forecast_timestamp = old_timestamp + timedelta(days=days_offset)
                
                base_load = float(row[load_col]) if pd.notna(row[load_col]) else 40000.0
                
                # Check if we have actual data for this timestamp
                actual_value = actual_loads.get(forecast_timestamp)
                
                predictions.append({
                    "timestamp": forecast_timestamp.isoformat(),
                    "predicted_load_mw": float(base_load),
                    "p10": float(base_load * 0.95),
                    "p50": float(base_load),
                    "p90": float(base_load * 1.05),
                    "actual_load_mw": actual_value,  # Include actual if available
                })
        
        # Ensure we have at least 96 predictions
        if len(predictions) < 96:
            base_load = 40000.0
            while len(predictions) < 96:
                timestamp = now + timedelta(minutes=len(predictions) * 15)
                predictions.append({
                    "timestamp": timestamp.isoformat(),
                    "predicted_load_mw": float(base_load),
                    "p10": float(base_load * 0.95),
                    "p50": float(base_load),
                    "p90": float(base_load * 1.05),
                    "actual_load_mw": None,
                })
        
        # Compute metrics from predictions and actuals
        pred_values = [p["p50"] for p in predictions if p["p50"]]
        actual_values = [p["actual_load_mw"] for p in predictions if p["actual_load_mw"]]
        
        if actual_values and pred_values:
            # Compute metrics comparing predictions to actuals
            mae = sum(abs(p["p50"] - p["actual_load_mw"]) for p in predictions if p["actual_load_mw"]) / len(actual_values)
            rmse = (sum((p["p50"] - p["actual_load_mw"]) ** 2 for p in predictions if p["actual_load_mw"]) / len(actual_values)) ** 0.5
            mape = (sum(abs(p["p50"] - p["actual_load_mw"]) / p["actual_load_mw"] * 100 for p in predictions if p["actual_load_mw"] and p["actual_load_mw"] > 0) / len(actual_values)) if actual_values else 0
        elif pred_values:
            mean_pred = sum(pred_values) / len(pred_values)
            mae = sum(abs(v - mean_pred) for v in pred_values) / len(pred_values)
            rmse = (sum((v - mean_pred) ** 2 for v in pred_values) / len(pred_values)) ** 0.5
            mape = 1.95
        else:
            mae, rmse, mape = 800.0, 1000.0, 1.95
        
        # Find peak
        peak_pred = max(predictions, key=lambda p: p["predicted_load_mw"])
        
        return {
            "historical": historical_data,
            "predictions": predictions,
            "metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
            },
            "peak": {
                "peak_value": round(peak_pred["predicted_load_mw"], 2),
                "peak_timestamp": peak_pred["timestamp"].replace("T", " ")[:16],
            },
            "forecast_date": now.strftime("%Y-%m-%d"),
            "history_end_date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "aggressiveness_pct": 0.0,
            "temperature_delta_c": None,
            "message": "Dummy data (inference failed) - Using historical pattern with actual data where available",
        }
    except Exception as e:
        logger.exception(f"Error in _generate_fallback_from_history: {e}")
        return _generate_minimal_fallback()


def _generate_minimal_fallback() -> dict:
    """Generate a minimal fallback response with constant values."""
    now = datetime.now()
    base_load = 41000.0
    predictions = []
    
    # Generate 24 hours of predictions
    for i in range(96):
        timestamp = now + timedelta(minutes=i * 15)
        predictions.append({
            "timestamp": timestamp.isoformat(),
            "predicted_load_mw": float(base_load),
            "p10": float(base_load * 0.95),
            "p50": float(base_load),
            "p90": float(base_load * 1.05),
            "actual_load_mw": None,
        })
    
    # Minimal historical data
    historical_data = []
    for i in range(288):
        timestamp = now - timedelta(days=3) + timedelta(minutes=i * 15)
        historical_data.append({
            "timestamp": timestamp.isoformat(),
            "load_mw": 40000.0,
        })
    
    return {
        "historical": historical_data,
        "predictions": predictions,
        "metrics": {
            "mae": 800.0,
            "rmse": 1000.0,
            "mape": 1.95,
        },
        "peak": {
            "peak_value": 43050.0,
            "peak_timestamp": now.strftime("%Y-%m-%d %H:%M"),
        },
        "forecast_date": now.strftime("%Y-%m-%d"),
        "history_end_date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "aggressiveness_pct": 0.0,
        "temperature_delta_c": None,
        "message": "Dummy data (inference failed) - Using constant fallback",
    }


