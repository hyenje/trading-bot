"""
앙상블 전략
여러 전략의 시그널을 종합하여 최종 매매 결정
- 가중 투표 방식
- 최소 합의 기준 충족 시에만 시그널 발생
"""
import logging
from typing import List, Dict
import pandas as pd

from strategies.base import BaseStrategy, Signal, TradeSignal
from strategies.ma_cross import MACrossStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_strategy import BollingerStrategy
from config import MAConfig, RSIConfig, BollingerConfig

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    복수 전략을 조합하여 최종 매매 시그널 생성
    (그리드 전략은 독립적으로 운영하므로 앙상블에서 제외)
    """

    def __init__(
        self,
        strategies: List[BaseStrategy] = None,
        weights: Dict[str, float] = None,
        min_agreement: float = 0.6,  # 최소 합의 비율
        min_confidence: float = 0.5,  # 최소 신뢰도
    ):
        super().__init__("Ensemble")

        self.strategies = strategies or [
            MACrossStrategy(MAConfig()),
            RSIStrategy(RSIConfig()),
            BollingerStrategy(BollingerConfig()),
        ]

        self.weights = weights or {
            "MA_Cross": 0.35,
            "RSI": 0.35,
            "Bollinger": 0.30,
        }

        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 전략의 지표를 합산"""
        result = df.copy()
        for strategy in self.strategies:
            result = strategy.get_indicators(result)
        return result

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        signals: List[TradeSignal] = []
        price = df["close"].iloc[-1]

        # 각 전략에서 시그널 수집
        for strategy in self.strategies:
            try:
                signal = strategy.analyze(df, symbol)
                signals.append(signal)
                logger.debug(
                    f"[{strategy.name}] {signal.signal.value} "
                    f"(신뢰도: {signal.confidence:.2f}) - {signal.reason}"
                )
            except Exception as e:
                logger.error(f"전략 분석 실패 [{strategy.name}]: {e}")

        if not signals:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=self.name,
                confidence=0.0,
                price=price,
                reason="전략 시그널 없음",
            )

        # 가중 투표 집계
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []

        for sig in signals:
            w = self.weights.get(sig.strategy_name, 1.0 / len(self.strategies))
            total_weight += w

            if sig.signal == Signal.BUY:
                buy_score += w * sig.confidence
                reasons.append(f"{sig.strategy_name}: 매수({sig.confidence:.2f})")
            elif sig.signal == Signal.SELL:
                sell_score += w * sig.confidence
                reasons.append(f"{sig.strategy_name}: 매도({sig.confidence:.2f})")
            else:
                reasons.append(f"{sig.strategy_name}: 관망")

        # 정규화
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        combined_reason = " | ".join(reasons)

        # 매수 합의
        buy_voters = sum(
            1 for s in signals if s.signal == Signal.BUY
        )
        sell_voters = sum(
            1 for s in signals if s.signal == Signal.SELL
        )
        total_voters = len(signals)

        if (
            buy_score > sell_score
            and buy_score >= self.min_confidence
            and buy_voters / total_voters >= self.min_agreement
        ):
            return TradeSignal(
                signal=Signal.BUY,
                symbol=symbol,
                strategy_name=self.name,
                confidence=buy_score,
                price=price,
                reason=f"앙상블 매수 ({buy_voters}/{total_voters} 합의) [{combined_reason}]",
                metadata={
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "voters": {s.strategy_name: s.signal.value for s in signals},
                },
            )

        if (
            sell_score > buy_score
            and sell_score >= self.min_confidence
            and sell_voters / total_voters >= self.min_agreement
        ):
            return TradeSignal(
                signal=Signal.SELL,
                symbol=symbol,
                strategy_name=self.name,
                confidence=sell_score,
                price=price,
                reason=f"앙상블 매도 ({sell_voters}/{total_voters} 합의) [{combined_reason}]",
                metadata={
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "voters": {s.strategy_name: s.signal.value for s in signals},
                },
            )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason=f"합의 미달 [{combined_reason}]",
        )
