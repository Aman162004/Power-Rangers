from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cached_session(cache_name: str, retry_total: int, backoff_factor: float):
    session = requests_cache.CachedSession(cache_name=cache_name, expire_after=timedelta(hours=6))
    retry = Retry(
        total=retry_total,
        connect=retry_total,
        read=retry_total,
        status=retry_total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _choose_openmeteo_url(start_dt: date, end_dt: date) -> str:
    today = datetime.now().date()
    if start_dt > today:
        return "https://api.open-meteo.com/v1/forecast"
    if end_dt > today:
        return "https://api.open-meteo.com/v1/forecast"
    return "https://archive-api.open-meteo.com/v1/archive"


def _iter_openmeteo_ranges(start_dt: date, end_dt: date) -> Iterable[tuple[str, date, date]]:
    today = datetime.now().date()

    if end_dt <= today:
        yield _choose_openmeteo_url(start_dt, end_dt), start_dt, end_dt
        return

    if start_dt > today:
        yield _choose_openmeteo_url(start_dt, end_dt), start_dt, end_dt
        return

    yield "https://archive-api.open-meteo.com/v1/archive", start_dt, today
    yield "https://api.open-meteo.com/v1/forecast", today + timedelta(days=1), end_dt


def _iter_date_chunks(start_dt: date, end_dt: date, chunk_days: int = 31) -> Iterable[tuple[date, date]]:
    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_openmeteo_weather_data(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Kolkata",
    retry_total: int = 5,
    backoff_factor: float = 0.3,
    cache_name: str | None = None,
) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo and upsample to 15-minute frequency."""

    if cache_name is None:
        cache_name = os.getenv("OPENMETEO_CACHE_NAME", "/tmp/power-rangers/openmeteo")

    cache_path = Path(cache_name)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    session = _cached_session(cache_name=str(cache_path), retry_total=retry_total, backoff_factor=backoff_factor)

    params_base = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": timezone,
    }

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    frames: list[pd.DataFrame] = []

    for url, range_start, range_end in _iter_openmeteo_ranges(start_dt, end_dt):
        for chunk_start, chunk_end in _iter_date_chunks(range_start, range_end):
            params = {
                **params_base,
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
            }

            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()

            payload = response.json()
            hourly = payload.get("hourly", {})

            chunk_df = pd.DataFrame(
                {
                    "timestamp": hourly.get("time", []),
                    "temperature": hourly.get("temperature_2m", []),
                    "humidity": hourly.get("relative_humidity_2m", []),
                    "wind_speed": hourly.get("wind_speed_10m", []),
                    "rainfall": hourly.get("precipitation", []),
                }
            )

            if not chunk_df.empty:
                chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"], errors="coerce")
                chunk_df = chunk_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                frames.append(chunk_df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "wind_speed", "rainfall"])

    weather_df = pd.concat(frames, ignore_index=True)
    weather_df = weather_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    weather_df = (
        weather_df.set_index("timestamp")
        .resample("15min")
        .interpolate(method="linear")
        .reset_index()
    )
    return weather_df
