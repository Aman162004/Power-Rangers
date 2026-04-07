from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def _choose_openmeteo_url(end_date: str) -> str:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end_dt >= date.today() - timedelta(days=5):
        return "https://api.open-meteo.com/v1/forecast"
    return "https://archive-api.open-meteo.com/v1/archive"


def fetch_openmeteo_weather_data(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Kolkata",
    retry_total: int = 5,
    backoff_factor: float = 0.3,
    cache_name: str = ".cache/openmeteo",
) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo and upsample to 15-minute frequency."""

    session = _cached_session(cache_name=cache_name, retry_total=retry_total, backoff_factor=backoff_factor)
    url = _choose_openmeteo_url(end_date=end_date)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
        ],
        "timezone": timezone,
    }

    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    hourly = payload.get("hourly", {})

    weather_df = pd.DataFrame(
        {
            "timestamp": hourly.get("time", []),
            "temperature": hourly.get("temperature_2m", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "rainfall": hourly.get("precipitation", []),
        }
    )

    if weather_df.empty:
        return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "wind_speed", "rainfall"])

    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    weather_df = (
        weather_df.set_index("timestamp")
        .resample("15min")
        .interpolate(method="linear")
        .reset_index()
    )
    return weather_df
