"""
리스크 관리 모듈
- 포지션 사이즈 관리
- 일일 손실 한도 관리
- 손절/익절 관리
- 포지션 추적
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

from strategies.base import TradeSignal
from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    entry_price: float
    amount: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    order_id: Optional[str] = None


@dataclass
class DailyStats:
    date: date = field(default_factory=date.today)
    realized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0


class RiskManager:
    """리스크 관리자"""

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trade_history: List[Dict] = []
        self.daily_stats = DailyStats()
        self.initial_capital: float = 0.0

    def set_capital(self, capital: float):
        self.initial_capital = capital

    # ----------------------------------------------------------
    # 진입 검증
    # ----------------------------------------------------------
    def can_open_position(self, signal: TradeSignal) -> tuple[bool, str]:
        """포지션 진입 가능 여부 확인"""

        # 1. 이미 포지션 보유 중인 심볼인지
        if signal.symbol in self.positions:
            return False, f"{signal.symbol} 포지션 이미 보유 중"

        # 2. 최대 포지션 수 확인
        if len(self.positions) >= self.config.max_positions:
            return False, f"최대 포지션 수({self.config.max_positions}) 도달"

        # 3. 일일 손실 한도 확인
        self._refresh_daily_stats()
        max_daily_loss = self.initial_capital * (self.config.max_daily_loss_pct / 100)
        if self.daily_stats.realized_pnl < -max_daily_loss:
            return (
                False,
                f"일일 손실 한도 초과 (손실: {self.daily_stats.realized_pnl:.2f}, "
                f"한도: -{max_daily_loss:.2f})",
            )

        # 4. 최소 신뢰도 확인
        if signal.confidence < 0.5:
            return False, f"신뢰도 부족 ({signal.confidence:.2f} < 0.5)"

        return True, "진입 가능"

    # ----------------------------------------------------------
    # 포지션 관리
    # ----------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        strategy: str,
        order_id: str = None,
    ) -> Position:
        """포지션 등록"""
        sl = entry_price * (1 - self.config.stop_loss_pct / 100)
        tp = entry_price * (1 + self.config.take_profit_pct / 100)

        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            amount=amount,
            entry_time=datetime.now(),
            stop_loss=sl,
            take_profit=tp,
            strategy=strategy,
            order_id=order_id,
        )
        self.positions[symbol] = pos
        logger.info(
            f"포지션 등록: {symbol} | 진입: {entry_price:.2f} | "
            f"SL: {sl:.2f} | TP: {tp:.2f} | 수량: {amount:.6f}"
        )
        return pos

    def close_position(
        self, symbol: str, exit_price: float, reason: str
    ) -> Optional[Dict]:
        """포지션 청산"""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        pnl = (exit_price - pos.entry_price) * pos.amount
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100

        trade_record = {
            "symbol": symbol,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "amount": pos.amount,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": pos.entry_time,
            "exit_time": datetime.now(),
            "strategy": pos.strategy,
            "reason": reason,
            "side": "sell",
        }
        self.trade_history.append(trade_record)

        # 일일 통계 업데이트
        self.daily_stats.realized_pnl += pnl
        self.daily_stats.trades_count += 1
        if pnl > 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

        logger.info(
            f"포지션 청산: {symbol} | 사유: {reason} | "
            f"PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)"
        )
        return trade_record

    def check_stop_loss_take_profit(
        self, symbol: str, current_price: float
    ) -> Optional[str]:
        """현재가 기준 손절/익절 확인"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        if current_price <= pos.stop_loss:
            return "stop_loss"
        elif current_price >= pos.take_profit:
            return "take_profit"
        return None

    def calculate_order_amount(self, symbol: str, price: float) -> float:
        """주문 수량 계산 (Kelly Criterion 간이 적용)"""
        base_amount = self.config.order_amount / price

        # 최근 승률 기반 사이즈 조절
        recent = (
            self.trade_history[-20:]
            if len(self.trade_history) >= 20
            else self.trade_history
        )
        if recent:
            wins = sum(1 for t in recent if t["pnl"] > 0)
            win_rate = wins / len(recent)
            # 승률 높으면 약간 더 투자, 낮으면 줄임
            multiplier = 0.5 + win_rate  # 0.5 ~ 1.5
            base_amount *= min(1.5, max(0.5, multiplier))

        return base_amount

    # ----------------------------------------------------------
    # 유틸리티
    # ----------------------------------------------------------
    def _refresh_daily_stats(self):
        """일일 통계 날짜 확인 및 리셋"""
        today = date.today()
        if self.daily_stats.date != today:
            logger.info(f"일일 통계 리셋 ({self.daily_stats.date} → {today})")
            self.daily_stats = DailyStats(date=today)

    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        self._refresh_daily_stats()
        positions_detail = []

        for symbol, pos in self.positions.items():
            positions_detail.append(
                {
                    "symbol": symbol,
                    "entry_price": pos.entry_price,
                    "amount": pos.amount,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "strategy": pos.strategy,
                }
            )

        return {
            "open_positions": len(self.positions),
            "positions": positions_detail,
            "daily_pnl": self.daily_stats.realized_pnl,
            "daily_trades": self.daily_stats.trades_count,
            "daily_win_rate": (
                self.daily_stats.wins / self.daily_stats.trades_count * 100
                if self.daily_stats.trades_count > 0
                else 0
            ),
            "total_trades": len(self.trade_history),
        }
