import yaml
import os
import json
from datetime import datetime
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

    # Add TFT-required indexing columns in saved datasets.
    featured_df['time_idx'] = range(len(featured_df))
    featured_df['group_id'] = 0

    # Save processed dataset
    repository.save_dataset(featured_df, "featured_data")

    # Build time-series datasets
    train_ds, val_ds, test_ds = dataset_builder.build_datasets(featured_df)

    # Save splits
    train_df, val_df, test_df = dataset_builder.split_dataframe(featured_df)

    repository.save_dataset(train_df, "train_data")
    repository.save_dataset(val_df, "val_data")
    repository.save_dataset(test_df, "test_data")

    metadata_path = config['data'].get('prep_metadata_path', os.path.join(config['data']['processed_path'], 'prep_metadata.json'))
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    split_summary = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(split_df) == 0:
            split_summary[split_name] = {'rows': 0, 'start_timestamp': None, 'end_timestamp': None}
        else:
            split_summary[split_name] = {
                'rows': int(len(split_df)),
                'start_timestamp': str(split_df['timestamp'].iloc[0]),
                'end_timestamp': str(split_df['timestamp'].iloc[-1]),
            }

    prep_metadata = {
        'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'source_raw_path': config['data']['raw_path'],
        'preprocess_report': dataset_builder.last_preprocess_report,
        'feature_columns': [col for col in featured_df.columns if col != 'timestamp'],
        'split_summary': split_summary,
        'total_rows_featured': int(len(featured_df)),
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(prep_metadata, f, indent=2)

    print("Pipeline completed successfully.")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Prep metadata saved to: {metadata_path}")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    run_pipeline(config_path)