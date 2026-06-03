"""
BTC 추세 롱/숏 전략
- EMA 21/34 크로스와 느린 EMA 기울기로 추세 방향 확인
- RSI로 너무 약한 크로스 필터링
- 백테스트용 롱/숏 시그널 전략이며, 현 Spot 봇의 실거래 숏 주문에는 연결하지 않음
"""
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, TradeSignal
from config import BTCTrendLongShortConfig


class BTCTrendLongShortStrategy(BaseStrategy):
    def __init__(self, config: BTCTrendLongShortConfig = None):
        super().__init__("BTC_Trend_LongShort")
        self.config = config or BTCTrendLongShortConfig()

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cfg = self.config

        df["ema_fast"] = df["close"].ewm(span=cfg.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=cfg.slow_ema, adjust=False).mean()
        df["ema_slope"] = df["ema_slow"] - df["ema_slow"].shift(cfg.slope_period)

        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / cfg.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["trend_gap"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
        return df

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        df = self.get_indicators(df)
        cfg = self.config
        min_rows = max(cfg.slow_ema, cfg.rsi_period) + cfg.slope_period + 2
        price = df["close"].iloc[-1]

        if symbol != "BTC/USDT":
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=self.name,
                confidence=0.0,
                price=price,
                reason="BTC 전용 전략",
            )

        if len(df) < min_rows:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=self.name,
                confidence=0.0,
                price=price,
                reason="데이터 부족",
            )

        current = df.iloc[-1]
        prev = df.iloc[-2]

        cross_up = (
            prev["ema_fast"] <= prev["ema_slow"]
            and current["ema_fast"] > current["ema_slow"]
        )
        cross_down = (
            prev["ema_fast"] >= prev["ema_slow"]
            and current["ema_fast"] < current["ema_slow"]
        )

        bullish = current["ema_slope"] > 0 and current["rsi"] >= cfg.long_rsi_min
        bearish = current["ema_slope"] < 0 and current["rsi"] <= cfg.short_rsi_max

        confidence = self._confidence(current)

        if cross_up and bullish and confidence >= cfg.min_confidence:
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=(
                    f"BTC 롱: EMA{cfg.fast_ema}/{cfg.slow_ema} 상향 전환, "
                    f"RSI {current['rsi']:.1f}, slope {current['ema_slope']:.2f}"
                ),
                metadata=self._metadata(current, "long"),
            )

        if cross_down and bearish and confidence >= cfg.min_confidence:
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=(
                    f"BTC 숏: EMA{cfg.fast_ema}/{cfg.slow_ema} 하향 전환, "
                    f"RSI {current['rsi']:.1f}, slope {current['ema_slope']:.2f}"
                ),
                metadata=self._metadata(current, "short"),
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason=(
                f"전환 없음: RSI {current['rsi']:.1f}, "
                f"gap {current['trend_gap']:.3%}"
            ),
        )

    def _confidence(self, row: pd.Series) -> float:
        gap_score = min(0.2, abs(row["trend_gap"]) * 20)
        slope_score = min(0.1, abs(row["ema_slope"]) / row["close"] * 100)
        rsi_distance = abs(row["rsi"] - 50) / 50
        rsi_score = min(0.1, rsi_distance)
        return min(0.9, 0.6 + gap_score + slope_score + rsi_score)

    @staticmethod
    def _metadata(row: pd.Series, side: str) -> dict:
        return {
            "position_side": side,
            "ema_fast": row["ema_fast"],
            "ema_slow": row["ema_slow"],
            "ema_slope": row["ema_slope"],
            "rsi": row["rsi"],
            "trend_gap": row["trend_gap"],
        }
