import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ForecastVisualizer:
    def __init__(self, config: dict):
        self.config = config

    def plot_forecast_curve(self, historical: pd.DataFrame, forecast: pd.DataFrame):
        """Plot historical demand and forecast curve."""
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=historical['timestamp'],
            y=historical['load_mw'],
            mode='lines',
            name='Historical Demand',
            line=dict(color='blue')
        ))

        # Forecast
        forecast_start = historical['timestamp'].iloc[-1]
        forecast_times = pd.date_range(start=forecast_start, periods=len(forecast), freq='15min')[1:]

        fig.add_trace(go.Scatter(
            x=forecast_times,
            y=forecast['q0.5'],
            mode='lines',
            name='Forecast (P50)',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='Electricity Demand Forecast',
            xaxis_title='Time',
            yaxis_title='Load (MW)',
            template='plotly_white'
        )

        return fig

    def plot_probabilistic_bands(self, forecast: pd.DataFrame, start_time: pd.Timestamp):
        """Plot forecast with uncertainty bands."""
        times = pd.date_range(start=start_time, periods=len(forecast), freq='15min')

        fig = go.Figure()

        # P10-P90 band
        fig.add_trace(go.Scatter(
            x=times,
            y=forecast['q0.9'],
            mode='lines',
            name='P90',
            line=dict(color='lightgray'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=times,
            y=forecast['q0.1'],
            mode='lines',
            fill='tonexty',
            name='P10-P90 Band',
            line=dict(color='lightgray'),
            fillcolor='rgba(169, 169, 169, 0.3)'
        ))

        # P50
        fig.add_trace(go.Scatter(
            x=times,
            y=forecast['q0.5'],
            mode='lines',
            name='P50 Forecast',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='Probabilistic Forecast with Uncertainty Bands',
            xaxis_title='Time',
            yaxis_title='Load (MW)',
            template='plotly_white'
        )

        return fig

    def highlight_peaks(self, forecast: pd.DataFrame, peak_time: pd.Timestamp, peak_value: float, start_time: pd.Timestamp):
        """Highlight peak demand in forecast."""
        fig = self.plot_probabilistic_bands(forecast, start_time)

        # Add peak marker
        fig.add_trace(go.Scatter(
            x=[peak_time],
            y=[peak_value],
            mode='markers',
            name='Peak Demand',
            marker=dict(color='orange', size=12, symbol='star')
        ))

        return fig

    def plot_scenario_comparison(self, base_forecast: pd.DataFrame, scenario_forecasts: dict, start_time: pd.Timestamp):
        """Compare base forecast with scenario forecasts."""
        times = pd.date_range(start=start_time, periods=len(base_forecast), freq='15min')

        fig = go.Figure()

        # Base forecast
        fig.add_trace(go.Scatter(
            x=times,
            y=base_forecast['q0.5'],
            mode='lines',
            name='Base Forecast',
            line=dict(color='blue')
        ))

        # Scenario forecasts
        colors = ['red', 'green', 'orange', 'purple']
        for i, (name, forecast) in enumerate(scenario_forecasts.items()):
            fig.add_trace(go.Scatter(
                x=times,
                y=forecast['q0.5'],
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))

        fig.update_layout(
            title='Scenario Forecast Comparison',
            xaxis_title='Time',
            yaxis_title='Load (MW)',
            template='plotly_white'
        )

        return fig

# Example usage:
# visualizer = ForecastVisualizer(config)
# fig = visualizer.plot_forecast_curve(historical_data, forecast_data)
# fig.show()