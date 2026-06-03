"""
이동평균 크로스 전략
- 단기 이동평균이 장기 이동평균을 상향 돌파 → 매수
- 단기 이동평균이 장기 이동평균을 하향 돌파 → 매도
- MACD 보조 지표로 신뢰도 보정
"""
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, TradeSignal
from config import MAConfig

logger = logging.getLogger(__name__)


class MACrossStrategy(BaseStrategy):
    def __init__(self, config: MAConfig = None):
        super().__init__("MA_Cross")
        self.config = config or MAConfig()

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_short"] = df["close"].rolling(window=self.config.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.config.long_period).mean()

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=self.config.signal_period, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        df = self.get_indicators(df)

        if len(df) < self.config.long_period + 2:
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

        # 골든 크로스: 단기 MA가 장기 MA 상향 돌파
        golden_cross = (
            prev["sma_short"] <= prev["sma_long"]
            and current["sma_short"] > current["sma_long"]
        )

        # 데드 크로스: 단기 MA가 장기 MA 하향 돌파
        dead_cross = (
            prev["sma_short"] >= prev["sma_long"]
            and current["sma_short"] < current["sma_long"]
        )

        # MACD 보조 신호
        macd_bullish = current["macd_hist"] > 0
        macd_bearish = current["macd_hist"] < 0

        if golden_cross:
            confidence = 0.7 + (0.2 if macd_bullish else 0.0)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"골든크로스 (SMA{self.config.short_period} > SMA{self.config.long_period})"
                + (" + MACD 상승" if macd_bullish else ""),
                metadata={
                    "sma_short": current["sma_short"],
                    "sma_long": current["sma_long"],
                    "macd_hist": current["macd_hist"],
                },
            )
        elif dead_cross:
            confidence = 0.7 + (0.2 if macd_bearish else 0.0)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"데드크로스 (SMA{self.config.short_period} < SMA{self.config.long_period})"
                + (" + MACD 하락" if macd_bearish else ""),
                metadata={
                    "sma_short": current["sma_short"],
                    "sma_long": current["sma_long"],
                    "macd_hist": current["macd_hist"],
                },
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason="크로스 시그널 없음",
        )
