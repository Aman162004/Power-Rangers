import torch
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from src.forecast_repository import ForecastRepository
from src.tft_model import TFTModel
import yaml

class ForecastEngine:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.repository = ForecastRepository(self.config)
        self.model = self.load_model()

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            import yaml
            return yaml.safe_load(f)

    def load_model(self) -> TFTModel:
        model = TFTModel(self.config)
        epoch = self.repository.load_model_checkpoint("final_model", model, torch.optim.Adam(model.parameters()))
        model.eval()
        return model

    def generate_forecast(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecast for the next horizon."""
        # Prepare input data similar to training
        input_df = input_df.copy()
        input_df['time_idx'] = range(len(input_df))
        input_df['group_id'] = 0

        # Create dataset for prediction
        prediction_dataset = TimeSeriesDataSet(
            input_df,
            time_idx="time_idx",
            target="load_mw",
            group_ids=["group_id"],
            min_encoder_length=self.config['pipeline']['encoder_window'] // 2,
            max_encoder_length=self.config['pipeline']['encoder_window'],
            min_prediction_length=1,
            max_prediction_length=self.config['pipeline']['decoder_window'],
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=['hour', 'day_of_week', 'month'],
            time_varying_known_reals=['sin_hour', 'cos_hour', 'temperature', 'humidity', 'wind_speed', 'rainfall'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=['load_mw'] + [f'load_lag_{lag}' for lag in self.config['features']['lags']] + [f'rolling_mean_{w}' for w in self.config['features']['rolling_windows']],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create dataloader
        from torch.utils.data import DataLoader
        pred_dataloader = DataLoader(prediction_dataset, batch_size=1)

        # Predict
        predictions = []
        for batch in pred_dataloader:
            with torch.no_grad():
                pred = self.model.predict(batch, mode="prediction")
                predictions.append(pred)

        # Process predictions (assuming quantiles)
        quantiles = self.config['model']['quantiles']
        pred_df = pd.DataFrame()
        for i, q in enumerate(quantiles):
            pred_df[f'q{q}'] = predictions[0][0, :, i].numpy()

        return pred_df

    def recursive_forecast(self, initial_data: pd.DataFrame, steps: int) -> pd.DataFrame:
        """Generate recursive forecast by updating input with predictions."""
        current_data = initial_data.copy()
        forecasts = []

        for _ in range(0, steps, self.config['pipeline']['decoder_window']):
            pred = self.generate_forecast(current_data)
            forecasts.append(pred)
            # Update current_data with predictions (simplified)
            # In practice, need to append predicted values and update features

        return pd.concat(forecasts, ignore_index=True)