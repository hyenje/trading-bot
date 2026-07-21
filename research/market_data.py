"""Small, research-only market-data helpers."""

from __future__ import annotations

import pandas as pd
import requests


def fetch_yahoo_adjusted_close(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Fetch daily adjusted closes from Yahoo's public chart endpoint."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": _unix_seconds(start),
        "period2": _unix_seconds(end),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "source-signal-research/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")

    data = result[0]
    timestamps = data.get("timestamp") or []
    indicators = data.get("indicators", {})
    adjusted = indicators.get("adjclose") or []
    values = (adjusted[0].get("adjclose") or []) if adjusted else []
    if not timestamps or not values:
        raise RuntimeError(f"Yahoo adjusted-close data missing for {ticker}")

    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
    series = pd.Series(values, index=index, name=ticker, dtype="float64").dropna()
    if series.empty:
        raise RuntimeError(f"Yahoo adjusted-close data empty for {ticker}")
    return series.groupby(series.index).last()


def _unix_seconds(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())
