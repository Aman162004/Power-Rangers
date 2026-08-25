import yaml
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineer import FeatureEngineer
from src.dataset_builder import DatasetBuilder
from src.shared.artifact_repository import ForecastRepository
from src.data_quality import DataQualityReporter


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(config_path: str):
    """Main pipeline execution with comprehensive quality reporting."""
    config = load_config(config_path)

    # Initialize components
    feature_engineer = FeatureEngineer(config)
    repository = ForecastRepository(config)
    dataset_builder = DatasetBuilder(config, feature_engineer)
    quality_reporter = DataQualityReporter(config)

    # Load and preprocess data from HISTORICAL lane only (immutable training corpus)
    historical_merged_file = config['data']['historical_merged_file']
    if not os.path.exists(historical_merged_file):
        raise FileNotFoundError(
            f"Historical merged dataset not found at {historical_merged_file}. "
            "Training requires the locked historical corpus."
        )
    
    raw_df = dataset_builder.load_raw_data(historical_merged_file)
    # Stage 1: feature engineering only, no split.
    featured_df = feature_engineer.engineer_features(raw_df)
    repository.save_dataset(featured_df, "featured_data")

    # Stage 2: cleaning, imputation, and final preparation.
    cleaned_df = dataset_builder.preprocess_data(featured_df)
    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df['time_idx'] = range(len(cleaned_df))
    cleaned_df['group_id'] = 0

    repository.save_dataset(cleaned_df, "cleaned_data")

    # Save final splits after cleaning.
    train_df, val_df, test_df = dataset_builder.split_dataframe(cleaned_df)

    repository.save_dataset(train_df, "train_data")
    repository.save_dataset(val_df, "val_data")
    repository.save_dataset(test_df, "test_data")

    # Build TFT datasets only when split sizes are sufficient.
    train_ds = val_ds = test_ds = None
    min_required_train_rows = config['pipeline']['encoder_window'] + config['pipeline']['decoder_window']
    if len(train_df) >= min_required_train_rows and len(val_df) > 0 and len(test_df) > 0:
        try:
            train_ds, val_ds, test_ds = dataset_builder.build_datasets(cleaned_df)
        except Exception as exc:
            print(f"Skipping TFT dataset construction due to validation error: {exc}")
    else:
        print(
            "Skipping TFT dataset construction: insufficient rows "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
            f"required_train>={min_required_train_rows})."
        )

    # Generate comprehensive quality report
    full_quality_report = quality_reporter.generate_full_report(raw_df, featured_df, cleaned_df)
    full_quality_report['preprocess_summary'] = dataset_builder.last_preprocess_report

    metadata_path = config['data'].get('prep_metadata_path', os.path.join(config['data']['historical_final_processed_path'], 'prep_metadata.json'))
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
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_raw_path': config['data']['historical_merged_file'],
        'preprocess_report': dataset_builder.last_preprocess_report,
        'feature_columns': [col for col in cleaned_df.columns if col != 'timestamp'],
        'split_summary': split_summary,
        'total_rows_featured': int(len(cleaned_df)),
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(prep_metadata, f, indent=2, default=_json_default)

    # Save comprehensive quality report
    quality_report_path = os.path.join(os.path.dirname(metadata_path), 'data_quality_report.json')
    with open(quality_report_path, 'w', encoding='utf-8') as f:
        json.dump(full_quality_report, f, indent=2, default=_json_default)

    print("Pipeline completed successfully.")
    print(f"Prep metadata saved to: {metadata_path}")
    print(f"Quality report saved to: {quality_report_path}")
    if train_ds is not None:
        print(f"Train dataset size: {len(train_ds)}")
        print(f"Val dataset size: {len(val_ds)}")
        print(f"Test dataset size: {len(test_ds)}")
    
    # Print quality summary
    print("\n=== DATA QUALITY SUMMARY ===")
    if 'quality_checks' in dataset_builder.last_preprocess_report:
        checks = dataset_builder.last_preprocess_report['quality_checks']
        if 'frequency_validation' in checks:
            freq = checks['frequency_validation']
            print(f"Frequency validation: {'PASS' if freq.get('valid_cadence') else 'WARNING'}")
            if freq.get('missing_intervals_count', 0) > 0:
                print(f"  - Missing intervals: {freq['missing_intervals_count']}")
        if 'weather_sanity' in checks:
            weather = checks['weather_sanity']
            print(f"Weather sanity checks: {'PASS' if weather.get('weather_checks_passed') else 'VIOLATIONS'}")
            if weather.get('total_violations', 0) > 0:
                print(f"  - Total violations: {weather['total_violations']}")
        if 'outlier_treatment' in checks:
            outliers = checks['outlier_treatment']
            if 'glitches_detected' in outliers:
                n_corrected = outliers.get('glitches_corrected', outliers.get('glitches_detected', 0))
                n_midnight = outliers.get('glitches_at_midnight', -1)
                print(
                    f"Outlier (boundary-glitch) treatment: {outliers['glitches_detected']} glitches "
                    f"detected, {n_corrected} corrected via rolling-MAD linear interpolation; "
                    f"{n_midnight} occurred at 00:00:00 day-boundaries. "
                    f"(mad_window={outliers.get('mad_window', '?')}, "
                    f"mad_mult={outliers.get('mad_mult', '?')})"
                )
            elif 'outliers_detected' in outliers:
                # Back-compat: legacy `outliers_detected` schema (pre-Step 1).
                print(f"Outlier treatment: {outliers['outliers_detected']} outliers detected and clipped")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    run_pipeline(config_path)