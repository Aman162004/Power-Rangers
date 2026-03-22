import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: pd.Series, save_path: str = None):
        """Plot predicted vs actual demand."""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual', alpha=0.7)
        plt.plot(y_pred.index, y_pred.values, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.title('Predicted vs Actual Electricity Demand')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate_forecast(self, actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
        """Evaluate forecast against actual data."""
        # Assume forecast_df has 'q0.5' as point forecast
        y_true = actual_df['load_mw'].values
        y_pred = forecast_df['q0.5'].values[:len(y_true)]  # Match lengths

        metrics = self.calculate_metrics(y_true, y_pred)

        # Create time index for plotting
        time_index = pd.date_range(start=actual_df['timestamp'].iloc[0],
                                   periods=len(y_true), freq='15min')
        y_true_series = pd.Series(y_true, index=time_index)
        y_pred_series = pd.Series(y_pred, index=time_index)

        self.plot_predictions_vs_actual(y_true_series, y_pred_series)

        return metrics

# Example usage:
# evaluator = ModelEvaluator(config)
# metrics = evaluator.evaluate_forecast(actual_data, forecast_data)
# print(metrics)