"""
백테스팅 엔진
과거 데이터를 기반으로 전략의 수익률을 시뮬레이션
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from strategies.base import BaseStrategy, Signal
from config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float = 0.0
    amount: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy: str = ""
    reason_entry: str = ""
    reason_exit: str = ""


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    initial_capital: float = 0.0
    position_size_usdt: Optional[float] = None
    capital_deployed_per_trade: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.position_size_usdt: Optional[float] = None

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "BTC/USDT",
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        allow_short: bool = False,
        position_size_usdt: Optional[float] = None,
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            df: OHLCV DataFrame (전체 기간)
            strategy: 전략 인스턴스
            symbol: 심볼
            stop_loss_pct: 손절 %
            take_profit_pct: 익절 %
            allow_short: SELL 신호를 숏 진입으로 허용할지 여부
            position_size_usdt: None이면 전액, 숫자이면 고정 USDT 명목금액

        Returns:
            BacktestResult
        """
        self.capital = self.config.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.position_size_usdt = position_size_usdt

        # 최소 데이터 확보를 위해 50번째 봉부터 시작
        lookback = 50

        for i in range(lookback, len(df)):
            window = df.iloc[: i + 1]
            current_price = window["close"].iloc[-1]
            current_time = window.index[-1]

            # 포지션 보유 중 → 손절/익절 확인
            if self.position:
                entry = self.position.entry_price
                high = window["high"].iloc[-1]
                low = window["low"].iloc[-1]

                if self.position.side == "long":
                    sl_price = entry * (1 - stop_loss_pct / 100)
                    if low <= sl_price:
                        self._close_position(sl_price, current_time, "손절")
                        self._update_equity(current_price, current_time)
                        continue

                    tp_price = entry * (1 + take_profit_pct / 100)
                    if high >= tp_price:
                        self._close_position(tp_price, current_time, "익절")
                        self._update_equity(current_price, current_time)
                        continue
                else:
                    sl_price = entry * (1 + stop_loss_pct / 100)
                    if high >= sl_price:
                        self._close_position(sl_price, current_time, "숏 손절")
                        self._update_equity(current_price, current_time)
                        continue

                    tp_price = entry * (1 - take_profit_pct / 100)
                    if low <= tp_price:
                        self._close_position(tp_price, current_time, "숏 익절")
                        self._update_equity(current_price, current_time)
                        continue

            # 전략 분석
            try:
                signal = strategy.analyze(window, symbol)
            except Exception as e:
                logger.debug(f"분석 건너뜀 (index {i}): {e}")
                self._update_equity(current_price, current_time)
                continue

            # 매매 실행
            if signal.signal == Signal.BUY:
                if not self.position:
                    self._open_position(
                        current_price,
                        current_time,
                        symbol,
                        strategy.name,
                        signal.reason,
                        side="long",
                    )
                elif self.position.side == "short":
                    self._close_position(current_price, current_time, signal.reason)
            elif signal.signal == Signal.SELL:
                if self.position and self.position.side == "long":
                    self._close_position(current_price, current_time, signal.reason)
                elif not self.position and allow_short:
                    self._open_position(
                        current_price,
                        current_time,
                        symbol,
                        strategy.name,
                        signal.reason,
                        side="short",
                    )

            self._update_equity(current_price, current_time)

        # 미청산 포지션 처리
        if self.position:
            last_price = df["close"].iloc[-1]
            last_time = df.index[-1]
            self._close_position(last_price, last_time, "백테스트 종료")

        return self._compile_result()

    def _open_position(
        self,
        price: float,
        time: datetime,
        symbol: str,
        strategy: str,
        reason: str,
        side: str = "long",
    ):
        if side == "long":
            entry_price = price * (1 + self.config.commission_rate)
        else:
            entry_price = price * (1 - self.config.commission_rate)

        notional = self.capital
        if self.position_size_usdt is not None:
            notional = min(self.capital, self.position_size_usdt)
        if notional <= 0:
            return

        self.position = Trade(
            entry_time=time,
            exit_time=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=notional / entry_price,
            strategy=strategy,
            reason_entry=reason,
        )
        logger.debug(f"{side} 포지션 진입: {entry_price:.2f} at {time}")

    def _close_position(self, price: float, time: datetime, reason: str):
        if not self.position:
            return

        if self.position.side == "long":
            exit_price = price * (1 - self.config.commission_rate)
            pnl = (exit_price - self.position.entry_price) * self.position.amount
            pnl_pct = (
                (exit_price - self.position.entry_price)
                / self.position.entry_price
                * 100
            )
        else:
            exit_price = price * (1 + self.config.commission_rate)
            pnl = (self.position.entry_price - exit_price) * self.position.amount
            pnl_pct = (
                (self.position.entry_price - exit_price)
                / self.position.entry_price
                * 100
            )

        self.position.exit_time = time
        self.position.exit_price = exit_price
        self.position.pnl = pnl
        self.position.pnl_pct = pnl_pct
        self.position.reason_exit = reason

        self.capital += self.position.pnl
        self.trades.append(self.position)
        self.position = None

        logger.debug(
            f"포지션 청산: {exit_price:.2f} at {time} | "
            f"PnL: {self.trades[-1].pnl:.2f}"
        )

    def _update_equity(self, price: float, time: datetime):
        equity = self.capital
        if self.position:
            if self.position.side == "long":
                unrealized = (price - self.position.entry_price) * self.position.amount
            else:
                unrealized = (self.position.entry_price - price) * self.position.amount
            equity += unrealized
        self.equity_curve.append(equity)
        self.timestamps.append(time)

    def _compile_result(self) -> BacktestResult:
        result = BacktestResult()
        result.initial_capital = self.config.initial_capital
        result.position_size_usdt = self.position_size_usdt
        result.capital_deployed_per_trade = (
            self.config.initial_capital
            if self.position_size_usdt is None
            else self.position_size_usdt
        )
        result.trades = self.trades
        result.equity_curve = self.equity_curve
        result.timestamps = self.timestamps
        result.total_trades = len(self.trades)

        if not self.trades:
            return result

        pnls = [t.pnl for t in self.trades]
        result.winning_trades = sum(1 for p in pnls if p > 0)
        result.losing_trades = sum(1 for p in pnls if p <= 0)
        result.win_rate = (
            result.winning_trades / result.total_trades * 100
            if result.total_trades > 0
            else 0
        )

        result.total_pnl = sum(pnls)
        result.total_pnl_pct = result.total_pnl / self.config.initial_capital * 100
        result.avg_trade_pnl = np.mean(pnls) if pnls else 0
        result.best_trade = max(pnls) if pnls else 0
        result.worst_trade = min(pnls) if pnls else 0

        # 최대 낙폭 (Max Drawdown)
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd / self.config.initial_capital * 100

        # 프로핏 팩터
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # 샤프 비율 (연환산, 무위험 이자율 0 가정)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.config.initial_capital
            result.sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )

        return result

    @staticmethod
    def format_result(result: BacktestResult) -> str:
        """결과를 보기 좋게 포맷"""
        return f"""
╔══════════════════════════════════════════╗
║          백테스팅 결과 리포트            ║
╠══════════════════════════════════════════╣
║ 총 거래 횟수:    {result.total_trades:>8d}                ║
║ 승률:           {result.win_rate:>8.1f}%               ║
║ 승리:           {result.winning_trades:>8d}                ║
║ 패배:           {result.losing_trades:>8d}                ║
╠══════════════════════════════════════════╣
║ 초기 자본:      ${result.initial_capital:>10.2f}              ║
║ 거래당 명목금액:${result.capital_deployed_per_trade:>10.2f}              ║
╠══════════════════════════════════════════╣
║ 총 수익:       ${result.total_pnl:>10.2f}              ║
║ 수익률:        {result.total_pnl_pct:>9.2f}%              ║
║ 평균 거래 수익: ${result.avg_trade_pnl:>10.2f}              ║
║ 최고 수익 거래: ${result.best_trade:>10.2f}              ║
║ 최악 손실 거래: ${result.worst_trade:>10.2f}              ║
╠══════════════════════════════════════════╣
║ 최대 낙폭:     ${result.max_drawdown:>10.2f}              ║
║ 최대 낙폭(%):   {result.max_drawdown_pct:>8.2f}%              ║
║ 프로핏 팩터:    {result.profit_factor:>9.2f}               ║
║ 샤프 비율:      {result.sharpe_ratio:>9.2f}               ║
╚══════════════════════════════════════════╝
"""
