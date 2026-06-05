"""
BTC 4h regime + lower-timeframe RSI/Bollinger pullback strategy.

This strategy is backtest-only for now. It separates broad market state first,
then looks for local pullback or mean-reversion entries on the entry timeframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from config import BTCRegimePullbackConfig
from strategies.base import BaseStrategy, Signal, TradeSignal


TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
UNKNOWN = "UNKNOWN"


class BTCRegimePullbackStrategy(BaseStrategy):
    """4h trend/range classifier with 15m/1h RSI + BB entries."""

    def __init__(self, config: BTCRegimePullbackConfig = None):
        super().__init__("BTC_Regime_Pullback")
        self.config = config or BTCRegimePullbackConfig()

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        data = df.copy()
        data["rsi"] = _rsi(data["close"], cfg.rsi_period)
        middle = data["close"].rolling(cfg.bb_period).mean()
        std = data["close"].rolling(cfg.bb_period).std(ddof=0)
        data["bb_middle"] = middle
        data["bb_upper"] = middle + cfg.bb_std * std
        data["bb_lower"] = middle - cfg.bb_std * std

        regime = build_regime_state_series(data, cfg)
        for column in regime.columns:
            data[column] = regime[column]
        return data

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        indicators = self.get_indicators(df)
        return self.analyze_indicators(indicators, symbol)

    def analyze_indicators(self, indicators: pd.DataFrame, symbol: str) -> TradeSignal:
        price = float(indicators["close"].iloc[-1])
        if symbol != "BTC/USDT":
            return self._hold(symbol, price, "BTC 전용 전략")

        min_rows = max(self.config.bb_period, self.config.rsi_period) + 2
        if len(indicators) < min_rows:
            return self._hold(symbol, price, "데이터 부족")

        return self.analyze_rows(indicators.iloc[-1], indicators.iloc[-2], symbol)

    def analyze_rows(
        self,
        current: pd.Series,
        previous: pd.Series,
        symbol: str,
    ) -> TradeSignal:
        price = float(current["close"])
        regime_value = current.get("regime_state")
        regime = UNKNOWN if pd.isna(regime_value) else str(regime_value)
        if regime == UNKNOWN:
            return self._hold(symbol, price, "4h regime 데이터 부족")

        mode = self.config.mode
        trend_allowed = mode in {"combined", "trend"}
        range_allowed = mode in {"combined", "range"}

        if regime == TREND_UP and trend_allowed:
            if self._long_pullback(current, previous):
                return self._signal(symbol, Signal.BUY, current, "trend")
            return self._hold(symbol, price, f"{regime}: 롱 눌림 대기")

        if regime == TREND_DOWN and trend_allowed:
            if self._short_rally(current, previous):
                return self._signal(symbol, Signal.SELL, current, "trend")
            return self._hold(symbol, price, f"{regime}: 숏 반등 대기")

        if regime == RANGE and range_allowed:
            long_entry = self._range_long(current, previous)
            short_entry = self._range_short(current, previous)
            if long_entry and not short_entry:
                return self._signal(symbol, Signal.BUY, current, "range")
            if short_entry and not long_entry:
                return self._signal(symbol, Signal.SELL, current, "range")
            return self._hold(symbol, price, "RANGE: 평균회귀 대기")

        return self._hold(symbol, price, f"{regime}: {mode} 모드 진입 없음")

    def _long_pullback(self, current: pd.Series, previous: pd.Series) -> bool:
        reentered_lower = (
            previous["close"] <= previous["bb_lower"]
            and current["close"] > current["bb_lower"]
        )
        rsi_recovering = (
            previous["rsi"] <= self.config.pullback_rsi_long
            and current["rsi"] > previous["rsi"]
            and current["close"] <= current["bb_middle"]
        )
        return bool(reentered_lower or rsi_recovering)

    def _short_rally(self, current: pd.Series, previous: pd.Series) -> bool:
        reentered_upper = (
            previous["close"] >= previous["bb_upper"]
            and current["close"] < current["bb_upper"]
        )
        rsi_cooling = (
            previous["rsi"] >= self.config.pullback_rsi_short
            and current["rsi"] < previous["rsi"]
            and current["close"] >= current["bb_middle"]
        )
        return bool(reentered_upper or rsi_cooling)

    def _range_long(self, current: pd.Series, previous: pd.Series) -> bool:
        reentered_lower = (
            previous["close"] <= previous["bb_lower"]
            and current["close"] > current["bb_lower"]
        )
        oversold_band = (
            current["rsi"] <= self.config.range_rsi_long
            and current["close"] <= current["bb_lower"]
        )
        return bool(reentered_lower or oversold_band)

    def _range_short(self, current: pd.Series, previous: pd.Series) -> bool:
        reentered_upper = (
            previous["close"] >= previous["bb_upper"]
            and current["close"] < current["bb_upper"]
        )
        overbought_band = (
            current["rsi"] >= self.config.range_rsi_short
            and current["close"] >= current["bb_upper"]
        )
        return bool(reentered_upper or overbought_band)

    def _signal(
        self,
        symbol: str,
        signal: Signal,
        row: pd.Series,
        exit_profile: str,
    ) -> TradeSignal:
        side = "long" if signal == Signal.BUY else "short"
        metadata = self._metadata(row, side, exit_profile)
        return TradeSignal(
            signal=signal,
            symbol=symbol,
            strategy_name=self.name,
            confidence=self._confidence(row),
            price=float(row["close"]),
            reason=(
                f"{row['regime_state']} {side}: RSI {row['rsi']:.1f}, "
                f"BB {row['bb_lower']:.2f}/{row['bb_middle']:.2f}/{row['bb_upper']:.2f}"
            ),
            metadata=metadata,
        )

    def _hold(self, symbol: str, price: float, reason: str) -> TradeSignal:
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason=reason,
        )

    def _metadata(self, row: pd.Series, side: str, exit_profile: str) -> dict:
        if exit_profile == "range":
            stop_loss = self.config.range_stop_loss_pct
            take_profit = self.config.range_take_profit_pct
            max_hold = self.config.range_max_hold_bars
        else:
            stop_loss = self.config.trend_stop_loss_pct
            take_profit = self.config.trend_take_profit_pct
            max_hold = 0
        return {
            "position_side": side,
            "regime_state": row.get("regime_state", UNKNOWN),
            "entry_mode": self.config.mode,
            "exit_profile": exit_profile,
            "exit_stop_loss_pct": stop_loss,
            "exit_take_profit_pct": take_profit,
            "exit_max_hold_bars": max_hold,
            "rsi": row.get("rsi"),
            "bb_middle": row.get("bb_middle"),
            "bb_upper": row.get("bb_upper"),
            "bb_lower": row.get("bb_lower"),
            "regime_trend_gap": row.get("regime_trend_gap"),
            "regime_slope_norm": row.get("regime_slope_norm"),
        }

    def _confidence(self, row: pd.Series) -> float:
        rsi_distance = abs(float(row["rsi"]) - 50.0) / 50.0
        middle = float(row["bb_middle"])
        band_width = float(row["bb_upper"] - row["bb_lower"])
        band_score = 0.0
        if middle > 0:
            band_score = min(0.15, band_width / middle)
        return min(0.9, self.config.min_confidence + min(0.15, rsi_distance) + band_score)


def build_regime_state_series(
    df: pd.DataFrame,
    config: BTCRegimePullbackConfig,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=df.index)

    htf = (
        df.resample(config.regime_timeframe, label="left", closed="left", origin="epoch")
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
        return _empty_regime_frame(df.index)

    htf["ema_fast"] = htf["close"].ewm(span=config.fast_ema, adjust=False).mean()
    htf["ema_slow"] = htf["close"].ewm(span=config.slow_ema, adjust=False).mean()
    htf["ema_slope"] = htf["ema_slow"] - htf["ema_slow"].shift(config.slope_period)
    htf["regime_trend_gap"] = (htf["ema_fast"] - htf["ema_slow"]) / htf["ema_slow"]
    htf["regime_slope_norm"] = htf["ema_slope"] / htf["close"]
    htf["regime_state"] = htf.apply(lambda row: classify_regime(row, config), axis=1)

    regime = htf[["regime_state", "regime_trend_gap", "regime_slope_norm"]].copy()
    offset = to_offset(config.regime_timeframe)
    regime.index = regime.index + offset
    regime["regime_closed_at"] = regime.index
    return regime.reindex(df.index, method="ffill")


def classify_regime(row: pd.Series, config: BTCRegimePullbackConfig) -> str:
    values = [
        row.get("ema_fast"),
        row.get("ema_slow"),
        row.get("regime_trend_gap"),
        row.get("regime_slope_norm"),
    ]
    if any(pd.isna(value) for value in values):
        return UNKNOWN

    gap = float(row["regime_trend_gap"])
    slope_norm = float(row["regime_slope_norm"])
    if abs(gap) < config.regime_min_gap_pct:
        return RANGE
    if gap > 0 and slope_norm > 0:
        return TREND_UP
    if gap < 0 and slope_norm < 0:
        return TREND_DOWN
    return RANGE


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _empty_regime_frame(index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime_state": UNKNOWN,
            "regime_trend_gap": np.nan,
            "regime_slope_norm": np.nan,
            "regime_closed_at": None,
        },
        index=index,
    )
