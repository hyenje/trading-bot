"""
BTC 롱/숏 시그널 관찰 모드
실제 주문 없이 전략 판단만 대시보드에 제공한다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from config import (
    BTCTrendLongShortConfig,
    DRY_RUN,
    LONG_SHORT_REGIME_TIMEFRAME,
    LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
    LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE,
    LONG_SHORT_TIMEFRAME,
)
from exchange import BinanceExchange
from strategies import BTCTrendLongShortStrategy, Signal
from strategies.btc_mtf_regime import (
    apply_regime_gate,
    bias_from_row,
    closed_candles,
    compute_regime_payload,
    side_from_signal,
    timeframe_seconds,
)


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
        regime_df = self.exchange.fetch_ohlcv(
            self.symbol, LONG_SHORT_REGIME_TIMEFRAME, limit=240
        )
        regime_df = closed_candles(regime_df, LONG_SHORT_REGIME_TIMEFRAME)
        signal_payload = self._empty_signal_payload("데이터 없음")

        if not df.empty:
            signal_payload = self._build_signal_payload(df, regime_df)

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

    def _build_signal_payload(
        self,
        df: pd.DataFrame,
        regime_df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        latest_signal = self.strategy.analyze(df, self.symbol)
        indicators = self.strategy.get_indicators(df)
        latest = indicators.iloc[-1]
        bias = bias_from_row(latest)
        regime_payload = compute_regime_payload(
            regime_df if regime_df is not None else pd.DataFrame(),
            LONG_SHORT_REGIME_TIMEFRAME,
            self.strategy.config,
        )

        payload = {
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
            "recent_signals": self._recent_signals(
                df, regime_payload.get("regime_side")
            ),
            "entry_block_reason": "",
            "reverse_block_reason": "",
            "reverse_policy": (
                "profit_only"
                if LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE
                else "always"
            ),
        }
        return apply_regime_gate(
            payload,
            regime_payload,
            LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
        )

    def _recent_signals(
        self,
        df: pd.DataFrame,
        regime_side: str | None = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        config = self.strategy.config
        min_rows = max(config.slow_ema, config.rsi_period) + config.slope_period + 2
        start = max(min_rows, len(df) - 120)

        for i in range(start, len(df)):
            window = df.iloc[: i + 1]
            signal = self.strategy.analyze(window, self.symbol)
            if signal.signal == Signal.HOLD:
                continue
            side = self._side(signal.signal)
            if (
                LONG_SHORT_REQUIRE_REGIME_ALIGNMENT
                and regime_side
                and side != regime_side
            ):
                continue
            rows.append(
                {
                    "time": window.index[-1].isoformat(),
                    "signal": signal.signal.value,
                    "side": side,
                    "regime_side": regime_side,
                    "regime_aligned": not regime_side or side == regime_side,
                    "price": self._number(signal.price),
                    "confidence": self._number(signal.confidence),
                    "reason": signal.reason,
                }
            )

        return rows[-8:]

    @staticmethod
    def _bias(row: pd.Series) -> str:
        return bias_from_row(row)

    @staticmethod
    def _side(signal: Signal) -> str:
        return side_from_signal(signal)

    def _empty_signal_payload(self, reason: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": Signal.HOLD.value,
            "side": "HOLD",
            "raw_signal": Signal.HOLD.value,
            "raw_side": "HOLD",
            "bias": "NEUTRAL",
            "confidence": 0.0,
            "price": 0.0,
            "reason": reason,
            "updated_at": datetime.now().isoformat(),
            "regime_timeframe": LONG_SHORT_REGIME_TIMEFRAME,
            "regime_side": "NEUTRAL",
            "regime_closed_at": None,
            "regime_aligned": False,
            "regime_alignment_required": LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
            "entry_block_reason": "",
            "reverse_policy": (
                "profit_only"
                if LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE
                else "always"
            ),
            "reverse_block_reason": "",
            "recent_signals": [],
        }

    def _closed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        return closed_candles(df, self.timeframe)

    def _timeframe_seconds(self) -> int:
        return timeframe_seconds(self.timeframe)

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
