import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict
import logging
from pytorch_forecasting import TimeSeriesDataSet

logger = logging.getLogger(__name__)


def _fix_boundary_glitches(
    df: pd.DataFrame,
    col: str = "load_mw",
    mad_window: int = 96,
    mad_mult: float = 6.0,
    min_periods: int = 24,
    interpolate_limit: int = 2,
) -> Dict[str, int]:
    """Detect and correct boundary glitches in-place on `df[col]`.

    Glitches at day boundaries (typically the first 00:00 row after the daily
    scraper concatenates day N and day N+1) manifest as a single large jump in
    the load series. The original `_combine_daily_frames` cannot fix this at
    ingestion time because the corrupted row has a *plausible* value (not NaN).

    Detection: scale-adaptive threshold based on the rolling median absolute
    deviation (MAD) of the per-step load differences (`Δload_t`). We flag any
    `|Δload_t|` that exceeds `mad_mult × rolling_MAD(Δload, window=mad_window)`.
    MAD on diffs (not on the level) normalizes for local volatility: during
    load ramps the natural MAD is high, during flat overnight periods the MAD
    is low — and the threshold scales accordingly.

    Correction: replace the flagged `df[col]` cells with NaN, then linearly
    interpolate across up to `interpolate_limit` consecutive NaNs (a single
    midnight glitch falls inside this limit). Records the audit column
    `f"{col}_was_glitch_corrected"` (parallel to the existing
    `f"{col}_was_missing"`).

    Returns a dict with correction statistics, intended to be merged into the
    `last_preprocess_report['quality_checks']['outlier_treatment']` slot.
    """
    if col not in df.columns:
        return {"glitches_detected": 0, "glitches_corrected": 0, "skipped": 1}

    audit_col = f"{col}_was_glitch_corrected"
    work = df[col].astype(float)
    diff = work.diff()

    # Rolling MAD of diffs. Using `np.median(np.abs(x - np.median(x)))` per
    # window; `raw=True` gives us a numpy array in the apply callback which is
    # markedly faster than passing a pandas Series slice.
    rolling_mad = diff.rolling(mad_window, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True,
    )
    threshold = mad_mult * rolling_mad
    glitch_mask = diff.abs() > threshold

    # Drop the first row (NaN diff) explicitly — never a glitch by definition.
    glitch_mask.iloc[0] = False

    n_glitches = int(glitch_mask.fillna(False).sum())
    if n_glitches == 0:
        df[audit_col] = 0
        return {"glitches_detected": 0, "glitches_corrected": 0}

    # Replace flagged values with NaN, then linearly interpolate a short
    # bridge. Using `interpolate(limit=2)` so a 1-tick glitch (the dominant
    # boundary case) and at most 2 consecutive glitched ticks are corrected
    # without spilling into adjacent legitimate gaps.
    df.loc[glitch_mask.fillna(False), col] = np.nan
    df[col] = df[col].interpolate(method="linear", limit=interpolate_limit, limit_direction="both")

    # Compute how many flagged cells remain uncorrected (interpolate-limit
    # exhausted). Those should be surfaced in the report.
    remaining_nan_mask = df[col].isna() & glitch_mask.fillna(False)
    n_remaining = int(remaining_nan_mask.sum())
    df[audit_col] = glitch_mask.fillna(False).astype(int)
    if n_remaining > 0:
        # Fall back to forward-fill then backward-fill for the residual cells.
        df[col] = df[col].ffill().bfill()
        n_remaining_after = int(df[col].isna().sum())
        if n_remaining_after > 0:
            df[col] = df[col].fillna(df[col].median())

    # Distribution stats for the audit report — useful to confirm that the
    # flagged cells cluster at midnight as expected.
    flagged_idx = df.index[df[audit_col] == 1]
    midnight_mask = pd.Series(False, index=df.index)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        midnight_mask = (ts.dt.hour == 0) & (ts.dt.minute == 0)
        n_midnight = int((df[audit_col] == 1).mul(midnight_mask).sum())
    else:
        n_midnight = -1

    stats = {
        "glitches_detected": n_glitches,
        "glitches_corrected": n_glitches - n_remaining,
        "glitches_at_midnight": n_midnight,
        "mad_window": mad_window,
        "mad_mult": mad_mult,
    }
    logger.warning(
        "[preprocess] boundary glitch correction on %s: %d detected, %d corrected, %d at 00:00:00 (mad_window=%d, mad_mult=%.1f)",
        col, n_glitches, n_glitches - n_remaining, n_midnight, mad_window, mad_mult,
    )
    return stats


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, encoder_window: int, decoder_window: int, stride: int = 1):
        self.encoder_window = encoder_window
        self.decoder_window = decoder_window
        self.stride = stride

        # Assume target is load_mw, features are all except timestamp and load_mw
        self.target_col = 'load_mw'
        self.feature_cols = [col for col in data.columns if col not in ['timestamp', self.target_col]]

        # Convert to numpy
        self.features = data[self.feature_cols].values.astype(np.float32)
        self.targets = data[self.target_col].values.astype(np.float32)

        self.length = len(data) - encoder_window - decoder_window + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        encoder_end = start_idx + self.encoder_window
        decoder_end = encoder_end + self.decoder_window

        # Encoder input: features for encoder_window
        encoder_input = self.features[start_idx:encoder_end]
        # Decoder target: load_mw for decoder_window
        decoder_target = self.targets[encoder_end:decoder_end]

        return torch.tensor(encoder_input), torch.tensor(decoder_target)


class DatasetBuilder:
    def __init__(self, config: dict, feature_engineer):
        self.config = config
        self.feature_engineer = feature_engineer
        self.last_preprocess_report = {}

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw CSV data."""
        return pd.read_csv(file_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with timestamp normalization, boundary glitch correction, and transparent missing handling.

        Pipeline order:
          1. Coerce timestamps → drop NaT rows.
          2. Sort + drop duplicates on timestamp.
          3. Boundary glitch correction on `load_mw` (rolling-MAD test on `Δload`).
             Adds `load_mw_was_glitch_corrected` audit column.
          4. Mark `f"{col}_was_missing"` indicators for every value column.
          5. ffill().bfill() → per-column median/mode safety net.
        """
        if 'timestamp' not in df.columns:
            raise ValueError("Input data must contain a 'timestamp' column.")

        work_df = df.copy()
        work_df['timestamp'] = pd.to_datetime(work_df['timestamp'], errors='coerce')
        invalid_timestamp_rows = int(work_df['timestamp'].isna().sum())
        work_df = work_df.dropna(subset=['timestamp'])

        pre_dedup_rows = len(work_df)
        work_df = work_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
        duplicate_rows_removed = int(pre_dedup_rows - len(work_df))

        # === Step 3: Boundary glitch correction ===
        # Threshold parameters are tunable via the `data` section of the config
        # under `glitch_correction: {mad_window, mad_mult, interpolate_limit}`,
        # defaulting to the prompt's recommended values.
        gc_cfg = self.config.get('data', {}).get('glitch_correction', {})
        glitch_stats = _fix_boundary_glitches(
            work_df,
            col='load_mw',
            mad_window=int(gc_cfg.get('mad_window', 96)),
            mad_mult=float(gc_cfg.get('mad_mult', 6.0)),
            min_periods=int(gc_cfg.get('min_periods', 24)),
            interpolate_limit=int(gc_cfg.get('interpolate_limit', 2)),
        )

        # === Step 4: missing-flags + ffill/bfill ===
        # Add glitch audit to value_columns before computing missing indicators
        # so its audit metadata is preserved through the fill.
        value_columns = [col for col in work_df.columns if col != 'timestamp']
        missing_counts_before_fill = work_df[value_columns].isna().sum().to_dict()

        # Keep explicit missing indicators so downstream users can inspect data quality.
        for col in value_columns:
            work_df[f'{col}_was_missing'] = work_df[col].isna().astype(int)

        work_df[value_columns] = work_df[value_columns].ffill().bfill()

        # Final safety net: fill any residual values using simple per-column statistics.
        for col in value_columns:
            if work_df[col].isna().any():
                if np.issubdtype(work_df[col].dtype, np.number):
                    work_df[col] = work_df[col].fillna(work_df[col].median())
                else:
                    mode_values = work_df[col].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
                    work_df[col] = work_df[col].fillna(fill_value)

        self.last_preprocess_report = {
            'rows_input': int(len(df)),
            'rows_after_timestamp_cleanup': int(len(work_df)),
            'invalid_timestamp_rows': invalid_timestamp_rows,
            'duplicate_rows_removed': duplicate_rows_removed,
            'missing_counts_before_fill': missing_counts_before_fill,
            'quality_checks': {
                # Populates the slot that `pipelines/prepare_historical_data.py`
                # already prints from — see `prep_metadata.json` and the
                # "[preprocess] Outlier treatment:" log line. Previously this
                # key was never written; now it carries real statistics.
                'outlier_treatment': glitch_stats,
            },
        }

        return work_df

    def split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe chronologically.

        Anchored on `data.test_start_date` (authoritative). Derives `val_start`
        and the implicit train cutoff backward from there:

            val_start = test_start_date - training_val_days
            train ends strictly before val_start
            val ends strictly before test_start (val → [val_start, test_start))
            test → [test_start, test_end_date + 23h45]

        This avoids the off-by-one that the previous forward-from-cutoff logic
        produced (which put val at the 50/50 midpoint of the future remainder,
        ~48 days, instead of the configured 45 days).

        Falls back to the 70/15/15 row-index split when `test_start_date` is
        not configured (e.g. smoke tests with synthetic data).
        """
        if 'timestamp' not in df.columns:
            raise ValueError("Input data must contain a 'timestamp' column.")

        work_df = df.copy()
        work_df['timestamp'] = pd.to_datetime(work_df['timestamp'], errors='coerce')
        work_df = work_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        data_cfg = self.config.get('data', {})
        test_start_str = data_cfg.get('test_start_date')
        val_days = int(data_cfg.get('training_val_days', 45))

        if test_start_str:
            test_start = pd.Timestamp(test_start_str)
            test_end_str = data_cfg.get('test_end_date')
            if test_end_str:
                test_end = pd.Timestamp(test_end_str)
            else:
                # Fall back to the last full day present in `work_df`.
                test_end = work_df['timestamp'].max().normalize()

            val_start = test_start - pd.Timedelta(days=val_days)

            train_df = work_df[work_df['timestamp'] < val_start].reset_index(drop=True)
            val_df = work_df[(work_df['timestamp'] >= val_start) & (work_df['timestamp'] < test_start)].reset_index(drop=True)
            # Test window inclusive of the final 23:45 timestamp of test_end_date.
            test_upper = test_end + pd.Timedelta(hours=23, minutes=45)
            test_df = work_df[(work_df['timestamp'] >= test_start) & (work_df['timestamp'] <= test_upper)].reset_index(drop=True)

            # === Loud assertions ===
            expected_test_rows = int(
                data_cfg.get('test_expected_rows', 96 * (test_end - test_start).days + 96)
            )
            assert not train_df.empty, (
                f"split_dataframe: train_df is empty (val_start={val_start}). "
                "Check that the historical data covers at least encoder_window + decoder_window rows before val_start."
            )
            assert not val_df.empty, (
                f"split_dataframe: val_df is empty (val_start={val_start}, test_start={test_start})."
            )
            assert not test_df.empty, (
                f"split_dataframe: test_df is empty (test_start={test_start}, test_end={test_end})."
            )
            assert len(test_df) == expected_test_rows, (
                f"split_dataframe: test_df has {len(test_df)} rows, expected {expected_test_rows} "
                f"(test_start={test_start.date()}, test_end={test_end.date()}, "
                f"coverage={len(test_df) / 96:.1f} days). "
                "This usually means the historical CSV does not extend to test_end_date — refusing to "
                "silently accept a near-miss because the test-window coverage directly biases walk-forward metrics."
            )
            if not val_df.empty and not test_df.empty:
                assert val_df['timestamp'].max() < test_start, (
                    f"split_dataframe: val end ({val_df['timestamp'].max()}) is not strictly before test_start ({test_start}). "
                    "A non-zero gap between val_end and test_start is required; overlapping splits would leak val data into test."
                )

            logger.info(
                "[split_dataframe] train=%d rows (last=%s), val=%d rows (%s → %s), test=%d rows (%s → %s)",
                len(train_df),
                train_df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if len(train_df) else "n/a",
                len(val_df),
                val_df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                val_df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
                len(test_df),
                test_df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                test_df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
            )

            return train_df, val_df, test_df

        # Legacy path: explicit `training_split_cutoff` (50/50 of remainder).
        # Used only when `test_start_date` is not set; emits a deprecation
        # warning so anyone still relying on the old semantics sees it.
        import warnings
        cutoff_date = data_cfg.get('training_split_cutoff')
        if cutoff_date:
            warnings.warn(
                "split_dataframe: `data.training_split_cutoff` is deprecated — set `data.test_start_date` "
                "and `data.training_val_days` instead. Falling back to 50/50-of-remainder logic which "
                "produces a ~48-day val/test each, not a 365-day holdout.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                cutoff_boundary = pd.Timestamp(cutoff_date) + pd.Timedelta(days=1)
            except Exception:
                cutoff_boundary = None

            if cutoff_boundary is not None:
                train_df = work_df[work_df['timestamp'] < cutoff_boundary].reset_index(drop=True)
                future_df = work_df[work_df['timestamp'] >= cutoff_boundary].reset_index(drop=True)

                if not train_df.empty and len(future_df) >= 2:
                    val_end = max(1, len(future_df) // 2)
                    val_df = future_df.iloc[:val_end].reset_index(drop=True)
                    test_df = future_df.iloc[val_end:].reset_index(drop=True)
                    if not val_df.empty and not test_df.empty:
                        return train_df, val_df, test_df

        n = len(work_df)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        train_df = work_df.iloc[:train_end].reset_index(drop=True)
        val_df = work_df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = work_df.iloc[val_end:].reset_index(drop=True)
        return train_df, val_df, test_df

    def build_datasets(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """Build train, val, test datasets using pytorch_forecasting."""
        lag_cols = [f'load_lag_{lag}' for lag in self.config['features']['lags']]
        rolling_cols = [f'rolling_mean_{w}' for w in self.config['features']['rolling_windows']]
        # Full set of columns referenced in the TimeSeriesDataSet construction below.
        # If any are missing, fall through to FeatureEngineer (which creates the
        # time-based + cyclical + lag/rolling columns; weather + is_holiday come
        # from the merged CSV in `load_raw_data`). Checking the full set — not just
        # hour/day_of_week/month — prevents a silent "use-as-is" path that would
        # later fail inside TimeSeriesDataSet with a cryptic KeyError.
        required_feature_cols = (['hour', 'day_of_week', 'month', 'sin_hour', 'cos_hour',
                                  'temperature', 'humidity', 'wind_speed', 'rainfall', 'is_holiday']
                                 + lag_cols + rolling_cols)

        if all(col in df.columns for col in required_feature_cols):
            df_featured = df.copy().reset_index(drop=True)
        else:
            df_featured = self.feature_engineer.engineer_features(df)
            df_featured = df_featured.dropna().reset_index(drop=True)

        # Add time_idx and group_id for TFT
        if 'time_idx' not in df_featured.columns:
            df_featured['time_idx'] = range(len(df_featured))
        if 'group_id' not in df_featured.columns:
            df_featured['group_id'] = 0  # Single series

        # Define features — must match the authoritative definition in
        # training_pipeline._build_tft_datasets and testing_pipeline.
        # hour/day_of_week/month are NUMERIC (int64) after FeatureEngineer and are
        # consumed as reals (StandardScaler) by the trained TFT. Declaring them as
        # categoricals here caused a dtype mismatch that silently broke TSD
        # construction (caught by prepare_historical_data's try/except).
        static_categoricals = []
        static_reals = []
        time_varying_known_categoricals = []
        time_varying_known_reals = ['hour', 'day_of_week', 'month', 'sin_hour', 'cos_hour',
                                    'temperature', 'humidity', 'wind_speed', 'rainfall', 'is_holiday']
        time_varying_unknown_categoricals = []
        time_varying_unknown_reals = ['load_mw'] + [f'load_lag_{lag}' for lag in self.config['features']['lags']] + [f'rolling_mean_{w}' for w in self.config['features']['rolling_windows']]

        # Split
        train_df, val_df, test_df = self.split_dataframe(df_featured)

        max_encoder_length = self.config['pipeline']['encoder_window']
        max_prediction_length = self.config['pipeline']['decoder_window']

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="load_mw",
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
        test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

        # === Scaler isolation check (Step 3) ===
        # Re-derived against the post-Step-2 boundaries (was originally verified
        # against the 2025-12-31 cutoff, which is now `test_start_date`=2025-04-07).
        #
        # Behavior of pytorch_forecasting `from_dataset`/`from_parameters`
        # (verified empirically on 2.5.1): the scaler OBJECTS are deep-copied,
        # but the deepcopy preserves the fitted statistics (mean_/scale_). The
        # functional contract we care about is therefore VALUE equality of the
        # fitted scaler state, not Python object identity. Compare numerically
        # and fail loudly if the val/test fitted state diverges from train's —
        # that would mean the val/test split recomputed its own normalization
        # (leaking its statistics into the encoder and contaminating losses).
        def _check_scaler_state(split_dataset, split_name: str) -> None:
            train_scalers = getattr(training, "_scalers", None)
            split_scalers = getattr(split_dataset, "_scalers", None)
            if train_scalers is None or split_scalers is None:
                logger.warning(
                    "[build_datasets] scaler isolation check skipped for %s "
                    "(dataset exposes no `_scalers` dict) — verify manually.",
                    split_name,
                )
                return
            for name, train_scaler in train_scalers.items():
                split_scaler = split_scalers.get(name)
                if split_scaler is None:
                    continue
                train_mean = getattr(train_scaler, "mean_", None)
                train_scale = getattr(train_scaler, "scale_", None)
                split_mean = getattr(split_scaler, "mean_", None)
                split_scale = getattr(split_scaler, "scale_", None)
                if train_mean is None or split_mean is None or train_scale is None or split_scale is None:
                    continue
                means_match = bool(np.allclose(np.asarray(train_mean), np.asarray(split_mean)))
                scales_match = bool(np.allclose(np.asarray(train_scale), np.asarray(split_scale)))
                assert means_match and scales_match, (
                    f"Scaler isolation violated for feature '{name}' on {split_name} split: "
                    f"fitted state diverges from train (mean_equal={means_match}, "
                    f"scale_equal={scales_match}). This means {split_name} rows are being "
                    "normalized against their own statistics instead of the train scaler."
                )

        _check_scaler_state(val_dataset, "val")
        _check_scaler_state(test_dataset, "test")
        logger.info(
            "[build_datasets] scaler isolation verified (val/test reuse the train-fitted "
            "scaler statistics against post-Step-2 boundaries)."
        )

        return training, val_dataset, test_dataset
