import streamlit as st
import pandas as pd
import numpy as np
from src.forecast_engine import ForecastEngine
from src.evaluation import ModelEvaluator
from src.streamlit_frontend.scenario_simulation import ScenarioSimulator
from src.streamlit_frontend.visualization import ForecastVisualizer
from src.streamlit_frontend.peak_detection import find_peak
from src.streamlit_frontend.probabilistic_forecasting import ProbabilisticForecasting
import yaml

def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")
    st.title("AI Electricity Demand Forecasting System")

    config = load_config()

    # Initialize components
    forecast_engine = ForecastEngine("config/config.yaml")
    evaluator = ModelEvaluator(config)
    simulator = ScenarioSimulator(config, forecast_engine)
    visualizer = ForecastVisualizer(config)
    pf = ProbabilisticForecasting(config)

    # Load sample data for demo
    try:
        historical_data = pd.read_parquet("data/processed/train_data.parquet").tail(100)
        input_data = historical_data.tail(24)  # Last 24 points for forecasting
    except:
        st.error("Please run the data pipeline first to generate processed data.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Forecast", "Evaluation", "Scenarios"])

    with tab1:
        st.header("Demand Forecast")

        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                forecast = forecast_engine.generate_forecast(input_data)

            # Display forecast
            st.subheader("Forecast Visualization")
            fig = visualizer.plot_forecast_curve(historical_data, forecast)
            st.plotly_chart(fig, use_container_width=True)

            # Probabilistic bands
            st.subheader("Uncertainty Bands")
            start_time = historical_data['timestamp'].iloc[-1]
            fig_bands = visualizer.plot_probabilistic_bands(forecast, start_time)
            st.plotly_chart(fig_bands, use_container_width=True)

            # Peak detection
            point_forecast = pf.get_point_forecast(forecast)
            timestamps = pd.date_range(start=start_time, periods=len(point_forecast), freq='15min')
            peak_info = find_peak(forecast)
            peak_time = pd.to_datetime(peak_info['peak_timestamp'])
            peak_value = peak_info['peak_value']

            st.subheader("Peak Demand Detection")
            st.write(f"Predicted Peak: {peak_value:.2f} MW at {peak_time}")

            fig_peak = visualizer.highlight_peaks(forecast, peak_time, peak_value, start_time)
            st.plotly_chart(fig_peak, use_container_width=True)

    with tab2:
        st.header("Model Evaluation")

        # Load test data
        try:
            test_data = pd.read_parquet("data/processed/test_data.parquet")
            forecast_test = forecast_engine.generate_forecast(test_data.tail(24))
            actual_test = test_data.tail(len(forecast_test))

            metrics = evaluator.evaluate_forecast(actual_test, forecast_test)

            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

            st.subheader("Predictions vs Actual")
            fig_eval = visualizer.plot_forecast_curve(actual_test, forecast_test)
            st.plotly_chart(fig_eval, use_container_width=True)

        except Exception as e:
            st.error(f"Evaluation data not available: {e}")

    with tab3:
        st.header("Scenario Simulation")

        st.subheader("Configure Scenario")
        temp_adjust = st.slider("Temperature Adjustment (°C)", -10.0, 10.0, 0.0)
        load_scale = st.slider("Load Scaling Factor", 0.5, 1.5, 1.0)

        if st.button("Run Scenario"):
            scenario_params = {
                'temperature_adjustment': temp_adjust,
                'load_scale': load_scale
            }

            with st.spinner("Running simulation..."):
                scenario_forecast = simulator.simulate_scenario(input_data, scenario_params)
                base_forecast = forecast_engine.generate_forecast(input_data)

            st.subheader("Scenario Comparison")
            start_time = historical_data['timestamp'].iloc[-1]
            scenario_forecasts = {'Scenario': scenario_forecast}
            fig_scenario = visualizer.plot_scenario_comparison(base_forecast, scenario_forecasts, start_time)
            st.plotly_chart(fig_scenario, use_container_width=True)

if __name__ == "__main__":
    main()