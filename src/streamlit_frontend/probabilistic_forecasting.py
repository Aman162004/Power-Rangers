import pandas as pd
import numpy as np

class ProbabilisticForecasting:
    def __init__(self, config: dict):
        self.config = config
        self.quantiles = config['model']['quantiles']

    def extract_quantiles(self, predictions_df: pd.DataFrame) -> dict:
        """Extract quantile predictions."""
        results = {}
        for q in self.quantiles:
            results[f'P{int(q*100)}'] = predictions_df[f'q{q}'].values
        return results

    def calculate_uncertainty(self, predictions_df: pd.DataFrame) -> pd.Series:
        """Calculate uncertainty as difference between P90 and P10."""
        p90 = predictions_df['q0.9']
        p10 = predictions_df['q0.1']
        return p90 - p10

    def get_point_forecast(self, predictions_df: pd.DataFrame) -> pd.Series:
        """Get point forecast as P50."""
        return predictions_df['q0.5']

# Example usage
if __name__ == "__main__":
    # Assuming predictions_df from forecast_engine
    pass