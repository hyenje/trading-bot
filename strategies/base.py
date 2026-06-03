"""
전략 베이스 클래스
모든 전략은 이 클래스를 상속받아 구현
"""
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    signal: Signal
    symbol: str
    strategy_name: str
    confidence: float  # 0.0 ~ 1.0
    price: float
    reason: str
    metadata: Optional[dict] = None


class BaseStrategy(ABC):
    """전략 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """
        데이터를 분석하여 매매 시그널을 반환
        Args:
            df: OHLCV DataFrame
            symbol: 거래 심볼
        Returns:
            TradeSignal
        """
        pass

    @abstractmethod
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표를 계산하여 DataFrame에 추가
        """
        pass
