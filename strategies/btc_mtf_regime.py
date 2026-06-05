"""
BTC 롱/숏 다중 시간봉 regime helper.

10분봉 전략 신호는 그대로 두고, 닫힌 상위 시간봉 추세로 진입 가능 여부만
판단한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from config import BTCTrendLongShortConfig
from strategies.base import Signal
from strategies.btc_trend_long_short import BTCTrendLongShortStrategy


def timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except (TypeError, ValueError):
        return 0
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 24 * 60 * 60
    return 0


def closed_candles(
    df: pd.DataFrame,
    timeframe: str,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    seconds = timeframe_seconds(timeframe)
    if seconds <= 0:
        return df

    latest = df.index[-1]
    if getattr(latest, "tzinfo", None):
        latest = latest.tz_convert(None)
    close_time = latest.to_pydatetime() + timedelta(seconds=seconds)
    if (now or datetime.utcnow()) < close_time:
        return df.iloc[:-1]
    return df


def side_from_signal(signal: Signal) -> str:
    if signal == Signal.BUY:
        return "LONG"
    if signal == Signal.SELL:
        return "SHORT"
    return "HOLD"


def bias_from_row(row: pd.Series) -> str:
    if row["ema_fast"] > row["ema_slow"] and row["ema_slope"] > 0:
        return "LONG_BIAS"
    if row["ema_fast"] < row["ema_slow"] and row["ema_slope"] < 0:
        return "SHORT_BIAS"
    return "NEUTRAL"


def regime_side_from_row(row: pd.Series) -> str:
    if row["ema_fast"] > row["ema_slow"] and row["ema_slope"] > 0:
        return "LONG"
    if row["ema_fast"] < row["ema_slow"] and row["ema_slope"] < 0:
        return "SHORT"
    return "NEUTRAL"


def compute_regime_payload(
    df: pd.DataFrame,
    timeframe: str,
    config: Optional[BTCTrendLongShortConfig] = None,
) -> Dict[str, Any]:
    cfg = config or BTCTrendLongShortConfig()
    empty = {
        "regime_timeframe": timeframe,
        "regime_side": "NEUTRAL",
        "regime_closed_at": None,
        "regime_ema_fast": 0.0,
        "regime_ema_slow": 0.0,
        "regime_ema_slope": 0.0,
        "regime_rsi": 0.0,
        "regime_trend_gap": 0.0,
    }
    min_rows = max(cfg.slow_ema, cfg.rsi_period) + cfg.slope_period + 1
    if df.empty or len(df) < min_rows:
        return empty

    strategy = BTCTrendLongShortStrategy(cfg)
    indicators = strategy.get_indicators(df)
    latest = indicators.iloc[-1]
    closed_at = indicators.index[-1] + to_offset(timeframe)
    return {
        "regime_timeframe": timeframe,
        "regime_side": regime_side_from_row(latest),
        "regime_closed_at": closed_at.isoformat(),
        "regime_ema_fast": _number(latest.get("ema_fast")),
        "regime_ema_slow": _number(latest.get("ema_slow")),
        "regime_ema_slope": _number(latest.get("ema_slope")),
        "regime_rsi": _number(latest.get("rsi")),
        "regime_trend_gap": _number(latest.get("trend_gap")),
    }


def apply_regime_gate(
    signal_payload: Dict[str, Any],
    regime_payload: Dict[str, Any],
    require_alignment: bool = True,
) -> Dict[str, Any]:
    payload = dict(signal_payload)
    raw_side = payload.get("side", "HOLD")
    payload.update(regime_payload)
    payload["raw_signal"] = payload.get("signal")
    payload["raw_side"] = raw_side
    payload["regime_alignment_required"] = require_alignment
    payload["regime_aligned"] = False
    payload.setdefault("entry_block_reason", "")

    if raw_side not in {"LONG", "SHORT"}:
        payload["regime_aligned"] = raw_side == payload.get("regime_side")
        return payload

    regime_side = payload.get("regime_side", "NEUTRAL")
    aligned = raw_side == regime_side
    payload["regime_aligned"] = aligned
    if require_alignment and not aligned:
        payload["side"] = "HOLD"
        payload["entry_block_reason"] = (
            f"{payload.get('regime_timeframe', '-')} regime mismatch: "
            f"raw {raw_side}, regime {regime_side}"
        )
    return payload


def build_regime_series(
    df: pd.DataFrame,
    timeframe: str,
    config: Optional[BTCTrendLongShortConfig] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)

    cfg = config or BTCTrendLongShortConfig()
    rule = timeframe
    htf = (
        df.resample(rule, label="left", closed="left", origin="epoch")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
    )
    if htf.empty:
        return pd.DataFrame(index=df.index)

    indicators = BTCTrendLongShortStrategy(cfg).get_indicators(htf)
    regime = pd.DataFrame(index=indicators.index)
    regime["regime_side"] = indicators.apply(regime_side_from_row, axis=1)
    regime["regime_ema_fast"] = indicators["ema_fast"]
    regime["regime_ema_slow"] = indicators["ema_slow"]
    regime["regime_ema_slope"] = indicators["ema_slope"]
    regime["regime_rsi"] = indicators["rsi"]
    regime["regime_trend_gap"] = indicators["trend_gap"]
    regime["regime_closed_at"] = indicators.index

    offset = to_offset(rule)
    regime.index = regime.index + offset
    regime["regime_closed_at"] = regime.index
    return regime.reindex(df.index, method="ffill")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
