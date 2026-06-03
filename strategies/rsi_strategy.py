"""
RSI (Relative Strength Index) 전략
- RSI가 과매도(30 이하) 구간 진입 후 반등 → 매수
- RSI가 과매수(70 이상) 구간 진입 후 하락 → 매도
- 거래량 확인으로 신뢰도 보정
"""
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, TradeSignal
from config import RSIConfig

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    def __init__(self, config: RSIConfig = None):
        super().__init__("RSI")
        self.config = config or RSIConfig()

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.config.period).mean()
        avg_loss = loss.rolling(window=self.config.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 거래량 이동평균
        df["vol_sma"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]

        return df

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        df = self.get_indicators(df)

        if len(df) < self.config.period + 5:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=self.name,
                confidence=0.0,
                price=df["close"].iloc[-1],
                reason="데이터 부족",
            )

        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current["close"]
        rsi = current["rsi"]
        prev_rsi = prev["rsi"]

        # 거래량 활성도 (신뢰도 보정 용)
        vol_active = current["vol_ratio"] > 1.0

        # 과매도 구간에서 반등 → 매수
        if prev_rsi < self.config.oversold and rsi >= self.config.oversold:
            confidence = 0.65 + (0.15 if vol_active else 0.0)
            # RSI가 낮을수록 더 강한 신호
            depth_bonus = min(0.15, (self.config.oversold - prev_rsi) / 100)
            confidence = min(1.0, confidence + depth_bonus)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"RSI 과매도 반등 ({prev_rsi:.1f} → {rsi:.1f})",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi, "vol_ratio": current["vol_ratio"]},
            )

        # 과매수 구간에서 하락 → 매도
        elif prev_rsi > self.config.overbought and rsi <= self.config.overbought:
            confidence = 0.65 + (0.15 if vol_active else 0.0)
            depth_bonus = min(0.15, (prev_rsi - self.config.overbought) / 100)
            confidence = min(1.0, confidence + depth_bonus)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"RSI 과매수 하락 ({prev_rsi:.1f} → {rsi:.1f})",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi, "vol_ratio": current["vol_ratio"]},
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason=f"RSI 중립 구간 ({rsi:.1f})",
        )
