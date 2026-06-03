"""
볼린저 밴드 전략
- 가격이 하단 밴드 이탈 후 복귀 → 매수
- 가격이 상단 밴드 이탈 후 복귀 → 매도
- 밴드 폭(bandwidth)으로 변동성 확인
"""
import pandas as pd
import numpy as np
import logging

from strategies.base import BaseStrategy, Signal, TradeSignal
from config import BollingerConfig

logger = logging.getLogger(__name__)


class BollingerStrategy(BaseStrategy):
    def __init__(self, config: BollingerConfig = None):
        super().__init__("Bollinger")
        self.config = config or BollingerConfig()

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["bb_mid"] = df["close"].rolling(window=self.config.period).mean()
        rolling_std = df["close"].rolling(window=self.config.period).std()
        df["bb_upper"] = df["bb_mid"] + (self.config.std_dev * rolling_std)
        df["bb_lower"] = df["bb_mid"] - (self.config.std_dev * rolling_std)

        # %B: 현재 가격이 밴드 내에서 어디에 위치하는지 (0~1)
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # 밴드 폭 (변동성 측정)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        return df

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        df = self.get_indicators(df)

        if len(df) < self.config.period + 2:
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

        # 밴드 폭이 좁으면 (스퀴즈) 신호 약화
        avg_width = df["bb_width"].rolling(window=50).mean().iloc[-1]
        is_squeeze = current["bb_width"] < avg_width * 0.8

        # 하단 밴드 이탈 후 복귀 → 매수
        if prev["close"] < prev["bb_lower"] and current["close"] >= current["bb_lower"]:
            confidence = 0.65
            if not is_squeeze:
                confidence += 0.15
            # 밴드 아래에서 멀리 떨어질수록 신뢰도 상승
            penetration = max(0, (prev["bb_lower"] - prev["close"]) / prev["bb_lower"])
            confidence = min(1.0, confidence + penetration * 5)
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"하단 밴드 복귀 (%B: {current['bb_pct_b']:.2f})",
                metadata={
                    "bb_upper": current["bb_upper"],
                    "bb_lower": current["bb_lower"],
                    "pct_b": current["bb_pct_b"],
                    "bandwidth": current["bb_width"],
                },
            )

        # 상단 밴드 이탈 후 복귀 → 매도
        elif prev["close"] > prev["bb_upper"] and current["close"] <= current["bb_upper"]:
            confidence = 0.65
            if not is_squeeze:
                confidence += 0.15
            penetration = max(0, (prev["close"] - prev["bb_upper"]) / prev["bb_upper"])
            confidence = min(1.0, confidence + penetration * 5)
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy_name=self.name,
                confidence=confidence,
                price=price,
                reason=f"상단 밴드 복귀 (%B: {current['bb_pct_b']:.2f})",
                metadata={
                    "bb_upper": current["bb_upper"],
                    "bb_lower": current["bb_lower"],
                    "pct_b": current["bb_pct_b"],
                    "bandwidth": current["bb_width"],
                },
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason=f"밴드 내 (%B: {current['bb_pct_b']:.2f})",
        )
