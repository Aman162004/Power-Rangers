import yaml
import os
from src.feature_engineer import FeatureEngineer
from src.dataset_builder import DatasetBuilder
from src.forecast_repository import ForecastRepository

def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(config_path: str):
    """Main pipeline execution."""
    config = load_config(config_path)

    # Initialize components
    feature_engineer = FeatureEngineer(config)
    repository = ForecastRepository(config)
    dataset_builder = DatasetBuilder(config, feature_engineer)

    # Load and preprocess data
    raw_df = dataset_builder.load_raw_data(config['data']['raw_path'])
    processed_df = dataset_builder.preprocess_data(raw_df)

    # Feature engineering
    featured_df = feature_engineer.engineer_features(processed_df)

    # Drop NaN
    featured_df = featured_df.dropna().reset_index(drop=True)

    # Save processed dataset
    repository.save_dataset(featured_df, "featured_data")

    # Build time-series datasets
    train_ds, val_ds, test_ds = dataset_builder.build_datasets(featured_df)

    # Save splits
    train_df = featured_df[:int(0.7 * len(featured_df))]
    val_df = featured_df[int(0.7 * len(featured_df)):int(0.85 * len(featured_df))]
    test_df = featured_df[int(0.85 * len(featured_df)):]

    repository.save_dataset(train_df, "train_data")
    repository.save_dataset(val_df, "val_data")
    repository.save_dataset(test_df, "test_data")

    print("Pipeline completed successfully.")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    run_pipeline(config_path)