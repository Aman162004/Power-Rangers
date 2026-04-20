from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOAD_CACHE_DIR = Path(os.getenv("SLDC_LOAD_CACHE_DIR", "/tmp/power-rangers/operational/raw"))
if not LOAD_CACHE_DIR.is_absolute():
    LOAD_CACHE_DIR = PROJECT_ROOT / LOAD_CACHE_DIR
LOAD_CACHE_PREFIX = "load_sldc"
ACTUAL_CACHE_PREFIX = "actual_sldc"
CURRENT_DAY_CACHE_TTL = timedelta(minutes=15)
ACTUAL_CACHE_FRESHNESS_WINDOW = timedelta(hours=1)


def _build_session(retry_total: int, backoff_factor: float, timeout_seconds: int) -> requests.Session:
    session = requests.Session()
    
    proxy_url = os.environ.get("SCRAPER_PROXY_URL")
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}

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
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )
    session.request_timeout = timeout_seconds
    return session


def _iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _extract_load_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    for table in tables:
        normalized_cols = [str(c).strip().upper() for c in table.columns]
        has_timeslot = "TIMESLOT" in normalized_cols
        has_delhi = "DELHI" in normalized_cols

        if has_timeslot and has_delhi:
            output = table.copy()
            output.columns = normalized_cols
            return output[["TIMESLOT", "DELHI"]]

        if not table.empty:
            header_candidate = [str(v).strip().upper() for v in table.iloc[0].tolist()]
            if "TIMESLOT" in header_candidate and "DELHI" in header_candidate:
                output = table.copy()
                output.columns = header_candidate
                output = output.iloc[1:].reset_index(drop=True)
                if "TIMESLOT" in output.columns and "DELHI" in output.columns:
                    return output[["TIMESLOT", "DELHI"]]

    return None


def _empty_load_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "load_mw"])


def _cache_path_for_day(day: date, cache_prefix: str = LOAD_CACHE_PREFIX) -> Path:
    return LOAD_CACHE_DIR / f"{cache_prefix}_{day.isoformat()}.csv"


def _normalize_day_frame(day_df: pd.DataFrame, day: date) -> pd.DataFrame:
    if day_df.empty:
        return _empty_load_frame()

    normalized = day_df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized["load_mw"] = pd.to_numeric(normalized["load_mw"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "load_mw"])
    if normalized.empty:
        return _empty_load_frame()

    normalized = normalized[normalized["timestamp"].dt.date == day]
    if normalized.empty:
        return _empty_load_frame()

    normalized = normalized.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return normalized[["timestamp", "load_mw"]]


def _cached_day_latest_timestamp(day_df: pd.DataFrame) -> datetime | None:
    if day_df.empty or "timestamp" not in day_df.columns:
        return None

    timestamps = pd.to_datetime(day_df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None

    latest_timestamp = timestamps.max()
    if pd.isna(latest_timestamp):
        return None

    return pd.Timestamp(latest_timestamp).to_pydatetime()


def _load_cached_day(
    day: date,
    cache_prefix: str = LOAD_CACHE_PREFIX,
    current_day_freshness_window: timedelta | None = None,
) -> pd.DataFrame | None:
    cache_path = _cache_path_for_day(day, cache_prefix)
    if not cache_path.exists():
        return None

    try:
        cached = pd.read_csv(cache_path)
    except Exception:
        return None

    if cached.empty or not {"timestamp", "load_mw"}.issubset(cached.columns):
        return None

    cached = _normalize_day_frame(cached, day)
    if cached.empty:
        return None

    if current_day_freshness_window is not None and day == date.today():
        latest_timestamp = _cached_day_latest_timestamp(cached)
        if latest_timestamp is None:
            return None

        if datetime.now() - latest_timestamp > current_day_freshness_window:
            return None

    if day == date.today() and current_day_freshness_window is None:
        try:
            modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime)
        except OSError:
            return None

        if datetime.now() - modified_at > CURRENT_DAY_CACHE_TTL:
            return None

    return cached


def _write_day_cache(day: date, day_df: pd.DataFrame, cache_prefix: str = LOAD_CACHE_PREFIX) -> None:
    cache_df = _normalize_day_frame(day_df, day)
    if cache_df.empty:
        return

    cache_path = _cache_path_for_day(day, cache_prefix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_df.to_csv(cache_path, index=False)
    except Exception:
        return


def _combine_daily_frames(daily_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not daily_frames:
        return _empty_load_frame()

    load_df = pd.concat(daily_frames, ignore_index=True)
    load_df = load_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])

    load_df = (
        load_df.set_index("timestamp")
        .resample("15min")
        .mean()
        .interpolate(method="linear")
        .reset_index()
    )
    return load_df


def _scrape_sldc_day(
    session: requests.Session,
    day: date,
    sleep_seconds: float,
) -> pd.DataFrame:
    date_param = day.strftime("%d/%m/%Y")
    url = f"https://www.delhisldc.org/Loaddata.aspx?mode={date_param}"

    response = session.get(url, timeout=session.request_timeout)
    if response.status_code != 200:
        time.sleep(sleep_seconds)
        return _empty_load_frame()

    soup = BeautifulSoup(response.text, "lxml")
    try:
        tables = pd.read_html(StringIO(str(soup)))
    except ValueError:
        time.sleep(sleep_seconds)
        return _empty_load_frame()

    extracted = _extract_load_table(tables)
    if extracted is None:
        time.sleep(sleep_seconds)
        return _empty_load_frame()

    extracted = extracted.copy()
    extracted["DELHI"] = pd.to_numeric(extracted["DELHI"], errors="coerce")
    extracted = extracted.dropna(subset=["DELHI", "TIMESLOT"])
    extracted["timestamp"] = pd.to_datetime(day.strftime("%Y-%m-%d") + " " + extracted["TIMESLOT"].astype(str), errors="coerce")
    extracted = extracted.dropna(subset=["timestamp"])

    day_df = extracted[["timestamp", "DELHI"]].rename(columns={"DELHI": "load_mw"})
    day_df = _normalize_day_frame(day_df, day)
    time.sleep(sleep_seconds)
    return day_df


def fetch_sldc_load_data(
    start_date: str,
    end_date: str,
    retry_total: int = 5,
    backoff_factor: float = 0.3,
    timeout_seconds: int = 10,
    sleep_seconds: float = 0.4,
    cache_prefix: str = LOAD_CACHE_PREFIX,
    current_day_freshness_window: timedelta | None = None,
) -> pd.DataFrame:
    """Fetch SLDC load data for a date range and normalize to 15-minute rows."""

    session = _build_session(retry_total=retry_total, backoff_factor=backoff_factor, timeout_seconds=timeout_seconds)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    daily_frames: list[pd.DataFrame] = []

    for day in _iter_dates(start_dt, end_dt):
        day_df = _load_cached_day(
            day,
            cache_prefix=cache_prefix,
            current_day_freshness_window=current_day_freshness_window,
        )
        if day_df is None:
            day_df = _scrape_sldc_day(session=session, day=day, sleep_seconds=sleep_seconds)
            _write_day_cache(day, day_df, cache_prefix=cache_prefix)

        if day_df.empty:
            continue

        daily_frames.append(day_df)

    return _combine_daily_frames(daily_frames)


def fetch_sldc_actual_load_data(
    start_date: str,
    end_date: str,
    retry_total: int = 5,
    backoff_factor: float = 0.3,
    timeout_seconds: int = 10,
    sleep_seconds: float = 0.4,
) -> pd.DataFrame:
    """Fetch SLDC data for chart actuals using a separate today cache."""

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end_dt < start_dt:
        return _empty_load_frame()

    today = date.today()
    historical_end = min(end_dt, today - timedelta(days=1))
    daily_frames: list[pd.DataFrame] = []

    if start_dt <= historical_end:
        historical_df = fetch_sldc_load_data(
            start_date=start_date,
            end_date=historical_end.isoformat(),
            retry_total=retry_total,
            backoff_factor=backoff_factor,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
        )
        if not historical_df.empty:
            daily_frames.append(historical_df)

    if start_dt <= today <= end_dt:
        today_str = today.isoformat()
        today_df = fetch_sldc_load_data(
            start_date=today_str,
            end_date=today_str,
            retry_total=retry_total,
            backoff_factor=backoff_factor,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            cache_prefix=ACTUAL_CACHE_PREFIX,
            current_day_freshness_window=ACTUAL_CACHE_FRESHNESS_WINDOW,
        )
        if today_df.empty:
            today_df = fetch_sldc_load_data(
                start_date=today_str,
                end_date=today_str,
                retry_total=retry_total,
                backoff_factor=backoff_factor,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
            )
        if not today_df.empty:
            daily_frames.append(today_df)

    return _combine_daily_frames(daily_frames)
