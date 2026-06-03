"""
BTC 롱/숏 시그널 관찰 모드
실제 주문 없이 전략 판단만 대시보드에 제공한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from config import BTCTrendLongShortConfig, DRY_RUN, LONG_SHORT_TIMEFRAME
from exchange import BinanceExchange
from strategies import BTCTrendLongShortStrategy, Signal


class BTCSignalObserver:
    """BTC 롱/숏 전략 신호를 조회 전용으로 관찰"""

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = LONG_SHORT_TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = BinanceExchange()
        self.strategy = BTCTrendLongShortStrategy(BTCTrendLongShortConfig())

    def get_status(self) -> Dict[str, Any]:
        df = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=240)
        df = self._closed_candles(df)
        signal_payload = self._empty_signal_payload("데이터 없음")

        if not df.empty:
            signal_payload = self._build_signal_payload(df)

        return {
            "running": True,
            "observer_mode": True,
            "dry_run": DRY_RUN,
            "balance": self.exchange.get_balance("USDT"),
            "daily_pnl": 0,
            "open_positions": 0,
            "daily_trades": 0,
            "daily_win_rate": 0,
            "total_trades": 0,
            "positions": [],
            "recent_trades": [],
            "equity_curve": [],
            "long_short_signal": signal_payload,
            "timestamp": datetime.now().isoformat(),
        }

    def _build_signal_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_signal = self.strategy.analyze(df, self.symbol)
        indicators = self.strategy.get_indicators(df)
        latest = indicators.iloc[-1]
        bias = self._bias(latest)

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": latest_signal.signal.value,
            "side": self._side(latest_signal.signal),
            "bias": bias,
            "confidence": self._number(latest_signal.confidence),
            "price": self._number(latest_signal.price),
            "reason": latest_signal.reason,
            "updated_at": df.index[-1].isoformat(),
            "ema_fast": self._number(latest.get("ema_fast")),
            "ema_slow": self._number(latest.get("ema_slow")),
            "ema_slope": self._number(latest.get("ema_slope")),
            "rsi": self._number(latest.get("rsi")),
            "trend_gap": self._number(latest.get("trend_gap")),
            "recent_signals": self._recent_signals(df),
        }

    def _recent_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        config = self.strategy.config
        min_rows = max(config.slow_ema, config.rsi_period) + config.slope_period + 2
        start = max(min_rows, len(df) - 120)

        for i in range(start, len(df)):
            window = df.iloc[: i + 1]
            signal = self.strategy.analyze(window, self.symbol)
            if signal.signal == Signal.HOLD:
                continue
            rows.append(
                {
                    "time": window.index[-1].isoformat(),
                    "signal": signal.signal.value,
                    "side": self._side(signal.signal),
                    "price": self._number(signal.price),
                    "confidence": self._number(signal.confidence),
                    "reason": signal.reason,
                }
            )

        return rows[-8:]

    @staticmethod
    def _bias(row: pd.Series) -> str:
        if row["ema_fast"] > row["ema_slow"] and row["ema_slope"] > 0:
            return "LONG_BIAS"
        if row["ema_fast"] < row["ema_slow"] and row["ema_slope"] < 0:
            return "SHORT_BIAS"
        return "NEUTRAL"

    @staticmethod
    def _side(signal: Signal) -> str:
        if signal == Signal.BUY:
            return "LONG"
        if signal == Signal.SELL:
            return "SHORT"
        return "HOLD"

    def _empty_signal_payload(self, reason: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": Signal.HOLD.value,
            "side": "HOLD",
            "bias": "NEUTRAL",
            "confidence": 0.0,
            "price": 0.0,
            "reason": reason,
            "updated_at": datetime.now().isoformat(),
            "recent_signals": [],
        }

    def _closed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        timeframe_seconds = self._timeframe_seconds()
        if timeframe_seconds <= 0:
            return df

        latest = df.index[-1]
        if getattr(latest, "tzinfo", None):
            latest = latest.tz_convert(None)
        close_time = latest.to_pydatetime() + timedelta(seconds=timeframe_seconds)
        if datetime.utcnow() < close_time:
            return df.iloc[:-1]
        return df

    def _timeframe_seconds(self) -> int:
        unit = self.timeframe[-1]
        try:
            value = int(self.timeframe[:-1])
        except ValueError:
            return 0
        if unit == "m":
            return value * 60
        if unit == "h":
            return value * 60 * 60
        if unit == "d":
            return value * 24 * 60 * 60
        return 0

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
