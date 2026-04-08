from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import holidays

from src.ingestion.load_fetcher import fetch_sldc_load_data
from src.ingestion.weather_fetcher import fetch_openmeteo_weather_data


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _default_window() -> tuple[str, str]:
    end = datetime.now().date()
    start = end - timedelta(days=30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_output_path(path_template: str, start_date: str, end_date: str) -> str:
    return path_template.format(start_date=start_date, end_date=end_date)


def _normalize_base_csv(base_csv_path: str) -> pd.DataFrame:
    base_df = pd.read_csv(base_csv_path)

    if "datetime" not in base_df.columns or "Power demand" not in base_df.columns:
        raise ValueError(
            "Base CSV must contain 'datetime' and 'Power demand' columns."
        )

    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(base_df["datetime"], errors="coerce"),
            "load_mw": pd.to_numeric(base_df["Power demand"], errors="coerce"),
        }
    )

    normalized = normalized.dropna(subset=["timestamp", "load_mw"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    normalized = (
        normalized.set_index("timestamp")
        .resample("15min")
        .mean()
        .interpolate(method="linear")
        .reset_index()
    )

    return normalized.reset_index(drop=True)


def _build_holiday_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    india_holidays = holidays.India(years=range(start_dt.year, end_dt.year + 1))

    rows = []
    current = start_dt.normalize()
    while current <= end_dt.normalize():
        rows.append(
            {
                "date": current.strftime("%Y-%m-%d"),
                "is_holiday": int(current.date() in india_holidays),
                "holiday_name": india_holidays.get(current.date(), ""),
            }
        )
        current += timedelta(days=1)

    return pd.DataFrame(rows)


def run_ingestion(config_path: str = "config/config.yaml") -> dict:
    """Fetch and persist raw load/weather snapshots and ingestion manifest."""

    config = _load_config(config_path)
    ingestion_cfg = config.get("ingestion", {})

    start_date = ingestion_cfg.get("start_date")
    end_date = ingestion_cfg.get("end_date")
    if not start_date or not end_date:
        start_date, end_date = _default_window()

    base_csv_path = ingestion_cfg.get("base_csv_path")

    latitude = float(ingestion_cfg.get("latitude", 28.6139))
    longitude = float(ingestion_cfg.get("longitude", 77.2090))
    timezone = ingestion_cfg.get("timezone", "Asia/Kolkata")

    retry_total = int(ingestion_cfg.get("retry_total", 5))
    backoff_factor = float(ingestion_cfg.get("backoff_factor", 0.3))
    timeout_seconds = int(ingestion_cfg.get("timeout_seconds", 10))
    sleep_seconds = float(ingestion_cfg.get("sldc_sleep_seconds", 0.4))

    outputs = ingestion_cfg.get("outputs", {})
    load_snapshot_template = outputs.get(
        "load_snapshot", "data/raw/load_sldc_{start_date}_to_{end_date}.csv"
    )
    weather_snapshot_template = outputs.get(
        "weather_snapshot", "data/raw/weather_openmeteo_{start_date}_to_{end_date}.csv"
    )
    merged_snapshot_template = outputs.get(
        "merged_snapshot", "data/raw/electricity_demand_{start_date}_to_{end_date}.csv"
    )
    manifest_template = outputs.get(
        "manifest", "data/raw/ingestion_manifest_{start_date}_to_{end_date}.json"
    )

    base_df = pd.DataFrame(columns=["timestamp", "load_mw"])
    scrape_start_date = start_date
    if base_csv_path:
        if not os.path.exists(base_csv_path):
            raise FileNotFoundError(f"Configured base_csv_path not found: {base_csv_path}")

        base_df = _normalize_base_csv(base_csv_path)
        if not base_df.empty:
            base_end = base_df["timestamp"].max().date()
            candidate_start = (base_end + timedelta(days=1)).strftime("%Y-%m-%d")
            scrape_start_date = max(start_date, candidate_start)

    scraped_load_df = pd.DataFrame(columns=["timestamp", "load_mw"])

    if scrape_start_date <= end_date:
        scraped_load_df = fetch_sldc_load_data(
            start_date=scrape_start_date,
            end_date=end_date,
            retry_total=retry_total,
            backoff_factor=backoff_factor,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
        )

    if scraped_load_df.empty and base_df.empty:
        raise RuntimeError("No ingestion data available from base CSV or SLDC scraping.")
    load_df = pd.concat([base_df, scraped_load_df], ignore_index=True)
    load_df = load_df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    overall_start = load_df["timestamp"].min().strftime("%Y-%m-%d")
    overall_end = load_df["timestamp"].max().strftime("%Y-%m-%d")

    weather_df = fetch_openmeteo_weather_data(
        start_date=overall_start,
        end_date=overall_end,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        retry_total=retry_total,
        backoff_factor=backoff_factor,
        cache_name=ingestion_cfg.get("weather_cache_path", ".cache/openmeteo"),
    )

    holiday_df = _build_holiday_calendar(overall_start, overall_end)

    merged_df = load_df.merge(weather_df, on="timestamp", how="left")
    merged_df['date'] = pd.to_datetime(merged_df['timestamp']).dt.strftime("%Y-%m-%d")
    merged_df = merged_df.merge(holiday_df[['date', 'is_holiday']], on='date', how='left')
    merged_df['is_holiday'] = merged_df['is_holiday'].fillna(0).astype(int)
    merged_df = merged_df.drop(columns=['date'])

    load_snapshot_path = _resolve_output_path(load_snapshot_template, overall_start, overall_end)
    weather_snapshot_path = _resolve_output_path(weather_snapshot_template, overall_start, overall_end)
    merged_snapshot_path = _resolve_output_path(merged_snapshot_template, overall_start, overall_end)
    manifest_path = _resolve_output_path(manifest_template, overall_start, overall_end)
    holiday_snapshot_path = os.path.join(
        config.get("data", {}).get("historical_calendar_path", "data/historical/raw/calendar/"),
        f"india_holidays_{overall_start}_to_{overall_end}.csv",
    )


    _write_csv(load_df, load_snapshot_path)
    _write_csv(weather_df, weather_snapshot_path)
    _write_csv(merged_df, merged_snapshot_path)
    _write_csv(holiday_df, holiday_snapshot_path)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = {
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "window": {
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "actual_start_date": overall_start,
            "actual_end_date": overall_end,
            "scrape_start_date": scrape_start_date,
        },
        "sources": {
            "sldc": "https://www.delhisldc.org/Loaddata.aspx",
            "open_meteo": "https://open-meteo.com/",
            "base_csv": base_csv_path,
        },
        "files": {
            "load_snapshot": {
                "path": load_snapshot_path,
                "rows": int(len(load_df)),
                "sha256": _sha256(load_snapshot_path),
            },
            "weather_snapshot": {
                "path": weather_snapshot_path,
                "rows": int(len(weather_df)),
                "sha256": _sha256(weather_snapshot_path),
            },
            "merged_snapshot": {
                "path": merged_snapshot_path,
                "rows": int(len(merged_df)),
                "sha256": _sha256(merged_snapshot_path),
            },
            "holiday_snapshot": {
                "path": holiday_snapshot_path,
                "rows": int(len(holiday_df)),
                "sha256": _sha256(holiday_snapshot_path),
            },
        },
        "quality": {
            "merged_missing_counts": merged_df.isna().sum().to_dict(),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest


if __name__ == "__main__":
    result = run_ingestion()
    print("Ingestion completed.")
    print(json.dumps(result["window"], indent=2))
