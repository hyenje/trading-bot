"""
그리드 트레이딩 전략
- 현재가 기준으로 상하 일정 범위에 그리드를 생성
- 가격이 그리드 레벨에 도달하면 매수/매도
- 횡보장에서 효과적
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

from strategies.base import BaseStrategy, Signal, TradeSignal
from config import GridConfig

logger = logging.getLogger(__name__)


class GridStrategy(BaseStrategy):
    def __init__(self, config: GridConfig = None):
        super().__init__("Grid")
        self.config = config or GridConfig()
        self.grid_levels: List[float] = []
        self.filled_levels: Dict[float, str] = {}  # level -> "buy" / "sell"
        self._initialized = False

    def setup_grid(self, center_price: float):
        """그리드 레벨 생성"""
        range_pct = self.config.grid_range_pct / 100
        lower = center_price * (1 - range_pct)
        upper = center_price * (1 + range_pct)
        step = (upper - lower) / self.config.grid_levels

        self.grid_levels = [lower + step * i for i in range(self.config.grid_levels + 1)]
        self.filled_levels = {}
        self._initialized = True

        logger.info(
            f"그리드 설정: 중심가={center_price:.2f}, "
            f"범위={lower:.2f}~{upper:.2f}, "
            f"레벨수={self.config.grid_levels}"
        )

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ATR (Average True Range) - 변동성 확인
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # 횡보장 판별용: ADX (간이 구현)
        df["price_change_pct"] = df["close"].pct_change(periods=20).abs() * 100
        return df

    def _find_nearest_grid(self, price: float, direction: str) -> float:
        """가격에 가장 가까운 그리드 레벨 찾기"""
        if direction == "below":
            candidates = [g for g in self.grid_levels if g <= price]
            return max(candidates) if candidates else 0
        else:
            candidates = [g for g in self.grid_levels if g >= price]
            return min(candidates) if candidates else 0

    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        df = self.get_indicators(df)
        price = df["close"].iloc[-1]

        if not self._initialized:
            self.setup_grid(price)

        # 그리드 범위 이탈 시 재설정
        if self.grid_levels:
            if price < self.grid_levels[0] * 0.95 or price > self.grid_levels[-1] * 1.05:
                logger.info("가격이 그리드 범위를 이탈하여 재설정합니다.")
                self.setup_grid(price)

        # 횡보장이 아닌 경우 (20봉 기준 변동 > 그리드 범위의 2배) → 보류
        recent_change = df["price_change_pct"].iloc[-1] if not pd.isna(df["price_change_pct"].iloc[-1]) else 0
        if recent_change > self.config.grid_range_pct * 2:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=self.name,
                confidence=0.0,
                price=price,
                reason=f"변동성 과다 ({recent_change:.1f}%), 그리드 부적합",
            )

        # 가격 아래 가장 가까운 그리드 (매수 레벨)
        buy_level = self._find_nearest_grid(price, "below")
        # 가격 위 가장 가까운 그리드 (매도 레벨)
        sell_level = self._find_nearest_grid(price, "above")

        prev_price = df["close"].iloc[-2]

        # 가격이 그리드 레벨을 하향 돌파 → 매수
        if buy_level and prev_price > buy_level and price <= buy_level:
            if buy_level not in self.filled_levels:
                self.filled_levels[buy_level] = "buy"
                return TradeSignal(
                    signal=Signal.BUY,
                    symbol=symbol,
                    strategy_name=self.name,
                    confidence=0.6,
                    price=price,
                    reason=f"그리드 매수 레벨 도달 ({buy_level:.2f})",
                    metadata={
                        "grid_level": buy_level,
                        "grid_count": len(self.grid_levels),
                        "filled": len(self.filled_levels),
                    },
                )

        # 가격이 그리드 레벨을 상향 돌파 → 매도
        if sell_level and prev_price < sell_level and price >= sell_level:
            # 이전에 매수했던 레벨이 있으면 매도
            lower_buys = [
                lv for lv, action in self.filled_levels.items()
                if action == "buy" and lv < sell_level
            ]
            if lower_buys:
                self.filled_levels[sell_level] = "sell"
                return TradeSignal(
                    signal=Signal.SELL,
                    symbol=symbol,
                    strategy_name=self.name,
                    confidence=0.6,
                    price=price,
                    reason=f"그리드 매도 레벨 도달 ({sell_level:.2f})",
                    metadata={
                        "grid_level": sell_level,
                        "buy_level": max(lower_buys),
                        "profit_pct": (sell_level - max(lower_buys)) / max(lower_buys) * 100,
                    },
                )

        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=price,
            reason="그리드 레벨 미도달",
        )
