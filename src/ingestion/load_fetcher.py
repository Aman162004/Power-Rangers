from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session(retry_total: int, backoff_factor: float, timeout_seconds: int) -> requests.Session:
    session = requests.Session()
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


def fetch_sldc_load_data(
    start_date: str,
    end_date: str,
    retry_total: int = 5,
    backoff_factor: float = 0.3,
    timeout_seconds: int = 10,
    sleep_seconds: float = 0.4,
) -> pd.DataFrame:
    """Fetch SLDC load data for a date range and normalize to 15-minute rows."""

    session = _build_session(retry_total=retry_total, backoff_factor=backoff_factor, timeout_seconds=timeout_seconds)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    daily_frames: list[pd.DataFrame] = []

    for day in _iter_dates(start_dt, end_dt):
        date_param = day.strftime("%d/%m/%Y")
        url = f"https://www.delhisldc.org/Loaddata.aspx?mode={date_param}"

        response = session.get(url, timeout=session.request_timeout)
        if response.status_code != 200:
            time.sleep(sleep_seconds)
            continue

        soup = BeautifulSoup(response.text, "lxml")
        try:
            tables = pd.read_html(str(soup))
        except ValueError:
            time.sleep(sleep_seconds)
            continue

        extracted = _extract_load_table(tables)
        if extracted is None:
            time.sleep(sleep_seconds)
            continue

        extracted = extracted.copy()
        extracted["DELHI"] = pd.to_numeric(extracted["DELHI"], errors="coerce")
        extracted = extracted.dropna(subset=["DELHI", "TIMESLOT"])
        extracted["timestamp"] = pd.to_datetime(day.strftime("%Y-%m-%d") + " " + extracted["TIMESLOT"].astype(str), errors="coerce")
        extracted = extracted.dropna(subset=["timestamp"])

        day_df = extracted[["timestamp", "DELHI"]].rename(columns={"DELHI": "load_mw"})
        daily_frames.append(day_df)
        time.sleep(sleep_seconds)

    if not daily_frames:
        return pd.DataFrame(columns=["timestamp", "load_mw"])

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
