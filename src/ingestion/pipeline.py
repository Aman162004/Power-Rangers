from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

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


def run_ingestion(config_path: str = "config/config.yaml") -> dict:
    """Fetch and persist raw load/weather snapshots and ingestion manifest."""

    config = _load_config(config_path)
    ingestion_cfg = config.get("ingestion", {})

    start_date = ingestion_cfg.get("start_date")
    end_date = ingestion_cfg.get("end_date")
    if not start_date or not end_date:
        start_date, end_date = _default_window()

    latitude = float(ingestion_cfg.get("latitude", 28.6139))
    longitude = float(ingestion_cfg.get("longitude", 77.2090))
    timezone = ingestion_cfg.get("timezone", "Asia/Kolkata")

    retry_total = int(ingestion_cfg.get("retry_total", 5))
    backoff_factor = float(ingestion_cfg.get("backoff_factor", 0.3))
    timeout_seconds = int(ingestion_cfg.get("timeout_seconds", 10))
    sleep_seconds = float(ingestion_cfg.get("sldc_sleep_seconds", 0.4))

    outputs = ingestion_cfg.get("outputs", {})
    load_snapshot_path = outputs.get("load_snapshot", "data/raw/load_sldc.csv")
    weather_snapshot_path = outputs.get("weather_snapshot", "data/raw/weather_openmeteo.csv")
    merged_snapshot_path = outputs.get("merged_snapshot", config["data"]["raw_path"])
    manifest_path = outputs.get("manifest", "data/raw/ingestion_manifest.json")

    load_df = fetch_sldc_load_data(
        start_date=start_date,
        end_date=end_date,
        retry_total=retry_total,
        backoff_factor=backoff_factor,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
    )

    weather_df = fetch_openmeteo_weather_data(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        retry_total=retry_total,
        backoff_factor=backoff_factor,
        cache_name=ingestion_cfg.get("weather_cache_path", ".cache/openmeteo"),
    )

    if load_df.empty:
        raise RuntimeError("SLDC load ingestion returned no rows for the selected date range.")

    merged_df = load_df.merge(weather_df, on="timestamp", how="left")
    merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

    _write_csv(load_df, load_snapshot_path)
    _write_csv(weather_df, weather_snapshot_path)
    _write_csv(merged_df, merged_snapshot_path)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = {
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "window": {"start_date": start_date, "end_date": end_date},
        "sources": {
            "sldc": "https://www.delhisldc.org/Loaddata.aspx",
            "open_meteo": "https://open-meteo.com/",
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
