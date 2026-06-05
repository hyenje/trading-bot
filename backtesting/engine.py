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

from config import BacktestConfig
from strategies.base import BaseStrategy, Signal
from strategies.btc_mtf_regime import build_regime_series, side_from_signal

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
    entry_market_price: float = 0.0
    exit_market_price: float = 0.0
    notional: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    fee_paid: float = 0.0
    gross_pnl: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_index: int = 0
    strategy: str = ""
    reason_entry: str = ""
    reason_exit: str = ""
    break_even_armed: bool = False
    exit_stop_loss_pct: Optional[float] = None
    exit_take_profit_pct: Optional[float] = None
    exit_max_hold_bars: Optional[int] = None


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
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
    flip_on_reverse: bool = False
    allowed_sides: str = "both"
    max_hold_bars: Optional[int] = None
    break_even_after_pct: Optional[float] = None
    commission_rate: float = 0.0
    slippage_rate: float = 0.0
    min_trend_gap: Optional[float] = None
    min_ema_slope: Optional[float] = None
    higher_timeframe: Optional[str] = None
    regime_timeframe: Optional[str] = None
    require_regime_alignment: bool = False
    reverse_only_when_profitable: bool = False
    min_reverse_net_pnl_usdt: float = 0.0
    blocked_by_regime_count: int = 0
    reverse_block_count: int = 0
    long_trades: int = 0
    short_trades: int = 0
    exit_reason_counts: Dict[str, int] = field(default_factory=dict)
    avg_hold_minutes: float = 0.0
    median_hold_minutes: float = 0.0
    max_hold_minutes: float = 0.0
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
        self.flip_on_reverse = False
        self.allowed_sides = "both"
        self.max_hold_bars: Optional[int] = None
        self.break_even_after_pct: Optional[float] = None
        self.min_trend_gap: Optional[float] = None
        self.min_ema_slope: Optional[float] = None
        self.higher_timeframe: Optional[str] = None
        self.regime_timeframe: Optional[str] = None
        self.require_regime_alignment = False
        self.reverse_only_when_profitable = False
        self.min_reverse_net_pnl_usdt = 0.0
        self.blocked_by_regime_count = 0
        self.reverse_block_count = 0

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "BTC/USDT",
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        allow_short: bool = False,
        position_size_usdt: Optional[float] = None,
        flip_on_reverse: bool = False,
        allowed_sides: str = "both",
        max_hold_bars: Optional[int] = None,
        break_even_after_pct: Optional[float] = None,
        min_trend_gap: Optional[float] = None,
        min_ema_slope: Optional[float] = None,
        higher_timeframe: Optional[str] = None,
        regime_timeframe: Optional[str] = None,
        require_regime_alignment: bool = False,
        reverse_only_when_profitable: bool = False,
        min_reverse_net_pnl_usdt: float = 0.0,
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
            flip_on_reverse: 반대 신호에서 청산 후 즉시 반대 포지션 진입
            allowed_sides: "both", "long", "short" 중 하나
            max_hold_bars: 설정 시 N개 봉 이상 보유하면 시간 청산
            break_even_after_pct: 설정 시 해당 수익률 도달 후 진입가 이탈 시 본전 청산
            min_trend_gap: 설정 시 절대 EMA gap이 이 값 미만이면 신규 진입 차단
            min_ema_slope: 설정 시 절대 EMA slope가 이 값 미만이면 신규 진입 차단
            higher_timeframe: 설정 시 해당 상위봉 EMA 방향과 같은 진입만 허용
            regime_timeframe: 설정 시 닫힌 상위봉 regime과 같은 신호만 허용
            require_regime_alignment: True이면 raw 신호와 regime 방향 일치 필요
            reverse_only_when_profitable: True이면 반대 신호 청산은 순손익 기준 통과 필요
            min_reverse_net_pnl_usdt: 반대 신호 청산/전환을 허용할 최소 순손익

        Returns:
            BacktestResult
        """
        self.capital = self.config.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.position_size_usdt = position_size_usdt
        self.flip_on_reverse = flip_on_reverse
        self.allowed_sides = allowed_sides
        self.max_hold_bars = max_hold_bars
        self.break_even_after_pct = break_even_after_pct
        self.min_trend_gap = min_trend_gap
        self.min_ema_slope = min_ema_slope
        self.higher_timeframe = higher_timeframe
        self.regime_timeframe = regime_timeframe
        self.require_regime_alignment = require_regime_alignment
        self.reverse_only_when_profitable = reverse_only_when_profitable
        self.min_reverse_net_pnl_usdt = min_reverse_net_pnl_usdt
        self.blocked_by_regime_count = 0
        self.reverse_block_count = 0

        # 최소 데이터 확보를 위해 50번째 봉부터 시작
        lookback = 50
        regime_series = None
        if regime_timeframe and require_regime_alignment:
            regime_series = build_regime_series(
                df,
                regime_timeframe,
                getattr(strategy, "config", None),
            )

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
                    trade_stop_loss_pct = self._position_stop_loss_pct(stop_loss_pct)
                    trade_take_profit_pct = self._position_take_profit_pct(take_profit_pct)
                    sl_price = entry * (1 - trade_stop_loss_pct / 100)
                    if not self.position.break_even_armed and low <= sl_price:
                        self._close_position(sl_price, current_time, "손절")
                        self._update_equity(current_price, current_time)
                        continue

                    if self._should_arm_break_even(high, low, entry):
                        self.position.break_even_armed = True

                    if self.position.break_even_armed and low <= entry:
                        self._close_position(entry, current_time, "본전 청산")
                        self._update_equity(current_price, current_time)
                        continue

                    tp_price = entry * (1 + trade_take_profit_pct / 100)
                    if high >= tp_price:
                        self._close_position(tp_price, current_time, "익절")
                        self._update_equity(current_price, current_time)
                        continue
                else:
                    trade_stop_loss_pct = self._position_stop_loss_pct(stop_loss_pct)
                    trade_take_profit_pct = self._position_take_profit_pct(take_profit_pct)
                    sl_price = entry * (1 + trade_stop_loss_pct / 100)
                    if not self.position.break_even_armed and high >= sl_price:
                        self._close_position(sl_price, current_time, "숏 손절")
                        self._update_equity(current_price, current_time)
                        continue

                    if self._should_arm_break_even(high, low, entry):
                        self.position.break_even_armed = True

                    if self.position.break_even_armed and high >= entry:
                        self._close_position(entry, current_time, "숏 본전 청산")
                        self._update_equity(current_price, current_time)
                        continue

                    tp_price = entry * (1 - trade_take_profit_pct / 100)
                    if low <= tp_price:
                        self._close_position(tp_price, current_time, "숏 익절")
                        self._update_equity(current_price, current_time)
                        continue

                if (
                    self._position_max_hold_bars(max_hold_bars) is not None
                    and i - self.position.entry_index
                    >= self._position_max_hold_bars(max_hold_bars)
                ):
                    self._close_position(current_price, current_time, "시간 청산")
                    self._update_equity(current_price, current_time)
                    continue

            # 전략 분석
            try:
                signal = strategy.analyze(window, symbol)
            except Exception as e:
                logger.debug(f"분석 건너뜀 (index {i}): {e}")
                self._update_equity(current_price, current_time)
                continue

            raw_side = side_from_signal(signal.signal)
            if not self._passes_regime_gate(raw_side, regime_series, current_time):
                self.blocked_by_regime_count += 1
                self._update_equity(current_price, current_time)
                continue

            # 매매 실행
            if signal.signal == Signal.BUY:
                if not self.position:
                    if self._can_open_side(strategy, window, "long", allow_short):
                        self._open_position(
                            current_price,
                            current_time,
                            i,
                            symbol,
                            strategy.name,
                            signal.reason,
                            side="long",
                            metadata=signal.metadata,
                        )
                elif self.position.side == "short":
                    if self._blocks_reverse_close(current_price):
                        self.reverse_block_count += 1
                        self._update_equity(current_price, current_time)
                        continue
                    self._close_position(current_price, current_time, signal.reason)
                    if flip_on_reverse and self._can_open_side(
                        strategy, window, "long", allow_short
                    ):
                        self._open_position(
                            current_price,
                            current_time,
                            i,
                            symbol,
                            strategy.name,
                            signal.reason,
                            side="long",
                            metadata=signal.metadata,
                        )
            elif signal.signal == Signal.SELL:
                if self.position and self.position.side == "long":
                    if self._blocks_reverse_close(current_price):
                        self.reverse_block_count += 1
                        self._update_equity(current_price, current_time)
                        continue
                    self._close_position(current_price, current_time, signal.reason)
                    if flip_on_reverse and self._can_open_side(
                        strategy, window, "short", allow_short
                    ):
                        self._open_position(
                            current_price,
                            current_time,
                            i,
                            symbol,
                            strategy.name,
                            signal.reason,
                            side="short",
                            metadata=signal.metadata,
                        )
                elif not self.position and self._can_open_side(
                    strategy, window, "short", allow_short
                ):
                    self._open_position(
                        current_price,
                        current_time,
                        i,
                        symbol,
                        strategy.name,
                        signal.reason,
                        side="short",
                        metadata=signal.metadata,
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
        index: int,
        symbol: str,
        strategy: str,
        reason: str,
        side: str = "long",
        metadata: Optional[dict] = None,
    ):
        entry_price = self._entry_fill_price(price, side)

        notional = self.capital
        if self.position_size_usdt is not None:
            notional = min(self.capital, self.position_size_usdt)
        if notional <= 0:
            return

        amount = notional / entry_price
        entry_fee = entry_price * amount * self.config.commission_rate
        self.position = Trade(
            entry_time=time,
            exit_time=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_market_price=price,
            amount=amount,
            notional=notional,
            entry_fee=entry_fee,
            entry_index=index,
            strategy=strategy,
            reason_entry=reason,
            exit_stop_loss_pct=self._metadata_float(metadata, "exit_stop_loss_pct"),
            exit_take_profit_pct=self._metadata_float(metadata, "exit_take_profit_pct"),
            exit_max_hold_bars=self._metadata_int(metadata, "exit_max_hold_bars"),
        )
        logger.debug(f"{side} 포지션 진입: {entry_price:.2f} at {time}")

    def _close_position(self, price: float, time: datetime, reason: str):
        if not self.position:
            return

        exit_price = self._exit_fill_price(price, self.position.side)
        if self.position.side == "long":
            gross_pnl = (exit_price - self.position.entry_price) * self.position.amount
        else:
            gross_pnl = (self.position.entry_price - exit_price) * self.position.amount
        exit_fee = exit_price * self.position.amount * self.config.commission_rate
        pnl = gross_pnl - self.position.entry_fee - exit_fee
        pnl_pct = pnl / self.position.notional * 100 if self.position.notional else 0.0

        self.position.exit_time = time
        self.position.exit_price = exit_price
        self.position.exit_market_price = price
        self.position.exit_fee = exit_fee
        self.position.fee_paid = self.position.entry_fee + exit_fee
        self.position.gross_pnl = gross_pnl
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
            equity += self._position_net_pnl_at(price)
        self.equity_curve.append(equity)
        self.timestamps.append(time)

    def _can_open_side(
        self,
        strategy: BaseStrategy,
        window: pd.DataFrame,
        side: str,
        allow_short: bool,
    ) -> bool:
        if side == "short" and not allow_short:
            return False
        if self.allowed_sides == "long" and side != "long":
            return False
        if self.allowed_sides == "short" and side != "short":
            return False
        if self.allowed_sides not in {"both", "long", "short"}:
            return False
        if not self._passes_indicator_filters(strategy, window):
            return False
        if not self._passes_higher_timeframe_filter(strategy, window, side):
            return False
        return True

    def _passes_indicator_filters(
        self, strategy: BaseStrategy, window: pd.DataFrame
    ) -> bool:
        if self.min_trend_gap is None and self.min_ema_slope is None:
            return True
        indicators = strategy.get_indicators(window)
        latest = indicators.iloc[-1]
        if self.min_trend_gap is not None:
            trend_gap = latest.get("trend_gap")
            if trend_gap is None or abs(float(trend_gap)) < self.min_trend_gap:
                return False
        if self.min_ema_slope is not None:
            ema_slope = latest.get("ema_slope")
            if ema_slope is None or abs(float(ema_slope)) < self.min_ema_slope:
                return False
        return True

    def _passes_higher_timeframe_filter(
        self,
        strategy: BaseStrategy,
        window: pd.DataFrame,
        side: str,
    ) -> bool:
        if not self.higher_timeframe:
            return True
        if "close" not in window:
            return False

        rule = self.higher_timeframe
        htf = (
            window.resample(rule, label="left", closed="left", origin="epoch")
            .agg({"close": "last"})
            .dropna()
        )
        if htf.empty:
            return False

        config = getattr(strategy, "config", None)
        fast_period = getattr(config, "fast_ema", 12)
        slow_period = getattr(config, "slow_ema", 26)
        if len(htf) < slow_period:
            return False

        fast = htf["close"].ewm(span=fast_period, adjust=False).mean().iloc[-1]
        slow = htf["close"].ewm(span=slow_period, adjust=False).mean().iloc[-1]
        if side == "long":
            return fast > slow
        return fast < slow

    def _passes_regime_gate(
        self,
        raw_side: str,
        regime_series: Optional[pd.DataFrame],
        current_time: datetime,
    ) -> bool:
        if not self.require_regime_alignment or raw_side not in {"LONG", "SHORT"}:
            return True
        if regime_series is None or regime_series.empty:
            return False
        try:
            regime_side = regime_series.loc[current_time, "regime_side"]
        except KeyError:
            return False
        if pd.isna(regime_side):
            return False
        return raw_side == str(regime_side)

    def _blocks_reverse_close(self, current_price: float) -> bool:
        if not self.reverse_only_when_profitable or not self.position:
            return False
        return self._position_net_pnl_at(current_price) < self.min_reverse_net_pnl_usdt

    def _should_arm_break_even(self, high: float, low: float, entry: float) -> bool:
        if (
            not self.position
            or self.position.break_even_armed
            or self.break_even_after_pct is None
            or self.break_even_after_pct <= 0
        ):
            return False
        if self.position.side == "long":
            return high >= entry * (1 + self.break_even_after_pct / 100)
        return low <= entry * (1 - self.break_even_after_pct / 100)

    def _position_stop_loss_pct(self, default: float) -> float:
        if self.position and self.position.exit_stop_loss_pct is not None:
            return self.position.exit_stop_loss_pct
        return default

    def _position_take_profit_pct(self, default: float) -> float:
        if self.position and self.position.exit_take_profit_pct is not None:
            return self.position.exit_take_profit_pct
        return default

    def _position_max_hold_bars(self, default: Optional[int]) -> Optional[int]:
        value = default
        if self.position and self.position.exit_max_hold_bars is not None:
            value = self.position.exit_max_hold_bars
        if value is None or value <= 0:
            return None
        return value

    @staticmethod
    def _metadata_float(metadata: Optional[dict], key: str) -> Optional[float]:
        if not metadata or metadata.get(key) in (None, ""):
            return None
        try:
            return float(metadata[key])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metadata_int(metadata: Optional[dict], key: str) -> Optional[int]:
        if not metadata or metadata.get(key) in (None, ""):
            return None
        try:
            return int(metadata[key])
        except (TypeError, ValueError):
            return None

    def _position_net_pnl_at(self, price: float) -> float:
        if not self.position:
            return 0.0
        exit_price = self._exit_fill_price(price, self.position.side)
        if self.position.side == "long":
            gross_pnl = (
                exit_price - self.position.entry_price
            ) * self.position.amount
        else:
            gross_pnl = (
                self.position.entry_price - exit_price
            ) * self.position.amount
        exit_fee = exit_price * self.position.amount * self.config.commission_rate
        return gross_pnl - self.position.entry_fee - exit_fee

    def _entry_fill_price(self, price: float, side: str) -> float:
        slippage = self.config.slippage_rate
        if side == "long":
            return price * (1 + slippage)
        return price * (1 - slippage)

    def _exit_fill_price(self, price: float, side: str) -> float:
        slippage = self.config.slippage_rate
        if side == "long":
            return price * (1 - slippage)
        return price * (1 + slippage)

    def _compile_result(self) -> BacktestResult:
        result = BacktestResult()
        result.initial_capital = self.config.initial_capital
        result.position_size_usdt = self.position_size_usdt
        result.flip_on_reverse = self.flip_on_reverse
        result.allowed_sides = self.allowed_sides
        result.max_hold_bars = self.max_hold_bars
        result.break_even_after_pct = self.break_even_after_pct
        result.commission_rate = self.config.commission_rate
        result.slippage_rate = self.config.slippage_rate
        result.min_trend_gap = self.min_trend_gap
        result.min_ema_slope = self.min_ema_slope
        result.higher_timeframe = self.higher_timeframe
        result.regime_timeframe = self.regime_timeframe
        result.require_regime_alignment = self.require_regime_alignment
        result.reverse_only_when_profitable = self.reverse_only_when_profitable
        result.min_reverse_net_pnl_usdt = self.min_reverse_net_pnl_usdt
        result.blocked_by_regime_count = self.blocked_by_regime_count
        result.reverse_block_count = self.reverse_block_count
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
        gross_pnls = [t.gross_pnl for t in self.trades]
        fees = [t.fee_paid for t in self.trades]
        result.winning_trades = sum(1 for p in pnls if p > 0)
        result.losing_trades = sum(1 for p in pnls if p <= 0)
        result.win_rate = (
            result.winning_trades / result.total_trades * 100
            if result.total_trades > 0
            else 0
        )

        result.total_pnl = sum(pnls)
        result.gross_pnl = sum(gross_pnls)
        result.total_fees = sum(fees)
        result.total_pnl_pct = result.total_pnl / self.config.initial_capital * 100
        result.avg_trade_pnl = np.mean(pnls) if pnls else 0
        result.best_trade = max(pnls) if pnls else 0
        result.worst_trade = min(pnls) if pnls else 0
        result.long_trades = sum(1 for trade in self.trades if trade.side == "long")
        result.short_trades = sum(1 for trade in self.trades if trade.side == "short")
        result.exit_reason_counts = self._exit_reason_counts()
        holds = self._hold_minutes()
        if holds:
            result.avg_hold_minutes = float(np.mean(holds))
            result.median_hold_minutes = float(np.median(holds))
            result.max_hold_minutes = max(holds)

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

    def _exit_reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trade in self.trades:
            key = self._exit_reason_key(trade.reason_exit)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _hold_minutes(self) -> List[float]:
        holds = []
        for trade in self.trades:
            if trade.exit_time is None:
                continue
            holds.append((trade.exit_time - trade.entry_time).total_seconds() / 60)
        return holds

    @staticmethod
    def _exit_reason_key(reason: str) -> str:
        if "손절" in reason:
            return "stop_loss"
        if "익절" in reason:
            return "take_profit"
        if "본전" in reason:
            return "break_even"
        if "시간" in reason:
            return "time_exit"
        if "종료" in reason:
            return "period_end"
        return "reverse_signal"

    @staticmethod
    def format_result(result: BacktestResult) -> str:
        """결과를 보기 좋게 포맷"""
        exit_counts = ", ".join(
            f"{key}:{value}" for key, value in result.exit_reason_counts.items()
        ) or "-"
        return f"""
╔══════════════════════════════════════════╗
║          백테스팅 결과 리포트            ║
╠══════════════════════════════════════════╣
║ 총 거래 횟수:    {result.total_trades:>8d}                ║
║ 롱 / 숏:        {result.long_trades:>4d} / {result.short_trades:<4d}             ║
║ 승률:           {result.win_rate:>8.1f}%               ║
║ 승리:           {result.winning_trades:>8d}                ║
║ 패배:           {result.losing_trades:>8d}                ║
╠══════════════════════════════════════════╣
║ 초기 자본:      ${result.initial_capital:>10.2f}              ║
║ 거래당 명목금액:${result.capital_deployed_per_trade:>10.2f}              ║
║ 수수료율:       {result.commission_rate * 10000:>8.1f}bp              ║
║ 슬리피지:       {result.slippage_rate * 10000:>8.1f}bp              ║
╠══════════════════════════════════════════╣
║ 총 수익(Net): ${result.total_pnl:>10.2f}              ║
║ 총 수익(Gross):${result.gross_pnl:>9.2f}              ║
║ 총 비용:       ${result.total_fees:>10.2f}              ║
║ 수익률:        {result.total_pnl_pct:>9.2f}%              ║
║ 평균 거래 수익: ${result.avg_trade_pnl:>10.2f}              ║
║ 최고 수익 거래: ${result.best_trade:>10.2f}              ║
║ 최악 손실 거래: ${result.worst_trade:>10.2f}              ║
╠══════════════════════════════════════════╣
║ 최대 낙폭:     ${result.max_drawdown:>10.2f}              ║
║ 최대 낙폭(%):   {result.max_drawdown_pct:>8.2f}%              ║
║ 프로핏 팩터:    {result.profit_factor:>9.2f}               ║
║ 샤프 비율:      {result.sharpe_ratio:>9.2f}               ║
╠══════════════════════════════════════════╣
║ Flip 진입:      {str(result.flip_on_reverse):>9s}               ║
║ 허용 방향:      {result.allowed_sides:>9s}               ║
║ Regime 차단:   {result.blocked_by_regime_count:>8d}                ║
║ Reverse 차단:  {result.reverse_block_count:>8d}                ║
║ 보유시간 중앙값:{result.median_hold_minutes:>8.1f}분              ║
║ 보유시간 평균:  {result.avg_hold_minutes:>8.1f}분              ║
║ 보유시간 최대:  {result.max_hold_minutes:>8.1f}분              ║
╚══════════════════════════════════════════╝
청산 사유: {exit_counts}
"""
