import pandas as pd
import numpy as np
from typing import Tuple

class PeakDemandDetector:
    def __init__(self, config: dict):
        self.config = config

    def detect_peak(self, predictions: pd.Series, timestamps: pd.Series) -> Tuple[pd.Timestamp, float]:
        """Detect the peak demand in the forecast horizon."""
        peak_idx = predictions.idxmax()
        peak_value = predictions.max()
        peak_timestamp = timestamps.iloc[peak_idx]
        return peak_timestamp, peak_value

    def detect_peaks_above_threshold(self, predictions: pd.Series, timestamps: pd.Series, threshold: float) -> pd.DataFrame:
        """Detect all peaks above a threshold."""
        peaks = predictions[predictions > threshold]
        return pd.DataFrame({
            'timestamp': timestamps.loc[peaks.index],
            'predicted_mw': peaks.values
        })

    def calculate_historical_threshold(self, historical_data: pd.Series, percentile: float = 95) -> float:
        """Calculate threshold based on historical data."""
        return np.percentile(historical_data, percentile)

# Example usage
if __name__ == "__main__":
    # Load historical data
    # predictions = get_point_forecast()
    # timestamps = pd.date_range(start='now', periods=len(predictions), freq='15min')
    # detector = PeakDemandDetector(config)
    # peak_time, peak_mw = detector.detect_peak(predictions, timestamps)
    # print(f"Peak at {peak_time}: {peak_mw} MW")
    pass