import pandas as pd
import numpy as np
from src.forecast_engine import ForecastEngine

class ScenarioSimulator:
    def __init__(self, config: dict, forecast_engine: ForecastEngine):
        self.config = config
        self.forecast_engine = forecast_engine

    def adjust_temperature(self, input_df: pd.DataFrame, adjustment: float) -> pd.DataFrame:
        """Adjust temperature by a fixed amount."""
        df = input_df.copy()
        df['temperature'] += adjustment
        return df

    def scale_load(self, input_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
        """Scale load by a factor."""
        df = input_df.copy()
        df['load_mw'] *= scale_factor
        return df

    def simulate_scenario(self, base_input: pd.DataFrame, scenario_params: dict) -> pd.DataFrame:
        """Run simulation with given parameters."""
        df = base_input.copy()

        if 'temperature_adjustment' in scenario_params:
            df = self.adjust_temperature(df, scenario_params['temperature_adjustment'])

        if 'load_scale' in scenario_params:
            df = self.scale_load(df, scenario_params['load_scale'])

        # Generate forecast with modified input
        forecast = self.forecast_engine.generate_forecast(df)
        return forecast

    def run_multiple_scenarios(self, base_input: pd.DataFrame, scenarios: list) -> dict:
        """Run multiple scenarios and return results."""
        results = {}
        for scenario in scenarios:
            name = scenario['name']
            params = scenario['params']
            forecast = self.simulate_scenario(base_input, params)
            results[name] = forecast
        return results

# Example usage:
# simulator = ScenarioSimulator(config, engine)
# scenario_params = {'temperature_adjustment': 5.0, 'load_scale': 1.1}
# forecast = simulator.simulate_scenario(input_data, scenario_params)