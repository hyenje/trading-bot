"""
BTC 롱/숏 Futures 테스트넷 실행 모드
실제 주문은 USE_TESTNET=true, DRY_RUN=false, ENABLE_LONG_SHORT_EXECUTION=true에서만 실행한다.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    BTCTrendLongShortConfig,
    DRY_RUN,
    LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES,
    LONG_SHORT_ENABLE_SIGNAL_CATCHUP,
    LONG_SHORT_LEVERAGE,
    LONG_SHORT_MAX_CONSECUTIVE_LOSSES,
    LONG_SHORT_MAX_DAILY_LOSS_PCT,
    LONG_SHORT_MAX_DAILY_LOSS_USDT,
    LONG_SHORT_MAX_DAILY_TRADES,
    LONG_SHORT_MAX_SIGNAL_AGE_MINUTES,
    LONG_SHORT_ORDER_USDT,
    LONG_SHORT_POLL_INTERVAL,
    LONG_SHORT_RISK_POLL_INTERVAL,
    LONG_SHORT_STOP_LOSS_PCT,
    LONG_SHORT_TAKE_PROFIT_PCT,
    LONG_SHORT_TIMEFRAME,
    mask_sensitive,
    validate_long_short_execution_allowed,
)
from exchange import BinanceFuturesExchange, ExchangeUnavailableError
from strategies import BTCTrendLongShortStrategy, Signal

logger = logging.getLogger(__name__)


class BTCLongShortExecutor:
    """BTC 롱/숏 전략 신호를 Futures 테스트넷 주문으로 실행"""

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = LONG_SHORT_TIMEFRAME):
        validate_long_short_execution_allowed()
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = BinanceFuturesExchange()
        if not self.exchange.check_private_access():
            raise RuntimeError(
                "Futures 테스트넷 인증에 실패했습니다. Futures 테스트넷 API Key/Secret을 확인하세요."
            )
        self.strategy = BTCTrendLongShortStrategy(BTCTrendLongShortConfig())
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._risk_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_signal_key: Optional[str] = None
        self._last_action = "대기"
        self._last_error = ""
        self._last_signal = self._empty_signal_payload("데이터 없음")
        self._last_position = self._flat_position()
        self._last_balance = 0.0
        self._last_protection_orders: List[Dict[str, Any]] = []
        self._market_state_status = "stale"
        self._last_market_error = ""
        self._last_market_update_at: Optional[str] = None
        self._entry_block_reason = "시장 상태 미확인"
        self._protection_reconciled = False
        self._safety_day = datetime.now().date()
        self._day_start_balance: Optional[float] = None
        self._daily_trade_count = 0
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._last_safety_reason = ""
        self.order_history: List[Dict[str, Any]] = []

        self.exchange.set_leverage(self.symbol, LONG_SHORT_LEVERAGE)

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._risk_thread = threading.Thread(target=self._risk_loop, daemon=True)
        self._risk_thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._risk_thread:
            self._risk_thread.join(timeout=5)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            signal_payload = dict(self._last_signal)
            order_history = list(self.order_history[-20:])
            last_action = self._last_action
            last_error = self._last_error
            position = dict(self._last_position)
            balance = self._last_balance
            protection_orders = list(self._last_protection_orders)
            market_state_status = self._market_state_status
            last_market_error = self._last_market_error
            last_market_update_at = self._last_market_update_at

        unrealized_pnl = position.get("unrealized_pnl", 0.0)
        safety_status = self._safety_status(position, balance)
        entry_block_reason = self._entry_block_reason_for(position)
        return {
            "running": self.running,
            "observer_mode": False,
            "execution_mode": True,
            "dry_run": DRY_RUN,
            "balance": balance,
            "daily_pnl": unrealized_pnl,
            "open_positions": 0 if position["side"] == "flat" else 1,
            "daily_trades": len(order_history),
            "daily_win_rate": 0,
            "total_trades": len(order_history),
            "positions": self._positions_payload(position),
            "recent_trades": [],
            "equity_curve": [],
            "long_short_signal": signal_payload,
            "executor_status": {
                "enabled": True,
                "market": "binance_futures_testnet",
                "timeframe": self.timeframe,
                "order_usdt": LONG_SHORT_ORDER_USDT,
                "leverage": LONG_SHORT_LEVERAGE,
                "poll_interval": LONG_SHORT_POLL_INTERVAL,
                "risk_poll_interval": LONG_SHORT_RISK_POLL_INTERVAL,
                "signal_catchup_enabled": LONG_SHORT_ENABLE_SIGNAL_CATCHUP,
                "stop_loss_pct": LONG_SHORT_STOP_LOSS_PCT,
                "take_profit_pct": LONG_SHORT_TAKE_PROFIT_PCT,
                "last_action": last_action,
                "last_error": last_error,
                "market_state_status": market_state_status,
                "entry_block_reason": entry_block_reason,
                "last_market_error": last_market_error,
                "last_market_update_at": last_market_update_at,
                "last_signal_key": self._last_signal_key,
                "protection_orders": protection_orders,
                "safety": safety_status,
                "orders": order_history,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"롱/숏 실행 루프 오류: {e}", exc_info=True)
                with self._lock:
                    self._last_error = mask_sensitive(e)
            time.sleep(LONG_SHORT_POLL_INTERVAL)

    def _risk_loop(self):
        while self.running:
            try:
                self._risk_tick()
            except Exception as e:
                logger.error(f"롱/숏 리스크 루프 오류: {e}", exc_info=True)
                with self._lock:
                    self._last_error = mask_sensitive(e)
            time.sleep(LONG_SHORT_RISK_POLL_INTERVAL)

    def _risk_tick(self):
        previous_position, previous_balance = self._cached_market_state()
        market_state = self._read_market_state("리스크 시장 상태 조회 실패")
        if not market_state:
            return
        position, balance = market_state
        self._record_external_position_close(
            previous_position,
            previous_balance,
            position,
            balance,
        )
        self._set_market_state(
            position=position,
            balance=balance,
        )

        if self._exit_for_safety(position, self._risk_signal_payload(position)):
            self._set_protection_orders([])
            return

        if self._exit_for_risk(position, self._risk_signal_payload(position)):
            self._set_protection_orders([])
            return

        if not self._should_skip_flat_protection_sync(position):
            self._sync_protection_orders(position)

    def _tick(self):
        try:
            df = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=240)
        except ExchangeUnavailableError as e:
            self._set_market_error(e, "시장 데이터 조회 실패")
            self._set_status(
                self._empty_signal_payload("시장 데이터 조회 실패"),
                "시장 데이터 조회 실패",
            )
            return
        df = self._closed_candles(df)
        if df.empty:
            self._set_status(self._empty_signal_payload("데이터 없음"), "데이터 없음")
            self._set_market_stale("캔들 데이터 없음")
            return

        signal_payload = self._build_signal_payload(df)
        self._set_status(signal_payload, "신호 관찰")
        market_state = self._read_market_state("시장 상태 조회 실패")
        if not market_state:
            self._set_status(signal_payload, "시장 상태 조회 실패")
            return
        position, balance = market_state
        self._set_market_state(position=position, balance=balance)
        if self._exit_for_safety(position, signal_payload):
            return
        if self._exit_for_risk(position, signal_payload):
            return

        signal_payload = self._signal_payload_for_entry(signal_payload, position)
        if signal_payload["side"] in {"LONG", "SHORT"}:
            action = "최근 신호 따라잡기" if signal_payload.get("catchup") else "신호 관찰"
            self._set_status(signal_payload, action)

        side = signal_payload["side"]
        if side not in {"LONG", "SHORT"}:
            return
        if self._is_stale(signal_payload["updated_at"]):
            self._set_status(signal_payload, "오래된 신호 무시")
            return

        signal_key = f"{signal_payload['updated_at']}:{side}"
        if signal_key == self._last_signal_key:
            return

        self._execute_side(side, signal_payload, position=position)
        self._last_signal_key = signal_key

    def _signal_payload_for_entry(
        self, signal_payload: Dict[str, Any], position: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not LONG_SHORT_ENABLE_SIGNAL_CATCHUP:
            return signal_payload
        if signal_payload.get("side") in {"LONG", "SHORT"}:
            return signal_payload
        if position.get("side") != "flat":
            return signal_payload

        recent_signal = self._latest_recent_signal(
            signal_payload.get("recent_signals", [])
        )
        if not recent_signal:
            return signal_payload

        recent_side = recent_signal.get("side")
        if recent_side not in {"LONG", "SHORT"}:
            return signal_payload
        if self._bias_side(signal_payload.get("bias")) != recent_side:
            return signal_payload
        if self._is_stale(recent_signal.get("time", "")):
            return signal_payload

        catchup_payload = dict(signal_payload)
        catchup_payload.update(
            {
                "signal": recent_signal.get("signal", signal_payload.get("signal")),
                "side": recent_side,
                "confidence": recent_signal.get(
                    "confidence", signal_payload.get("confidence", 0.0)
                ),
                "price": recent_signal.get("price", signal_payload.get("price", 0.0)),
                "reason": f"최근 신호 따라잡기: {recent_signal.get('reason', '-')}",
                "updated_at": recent_signal.get(
                    "time", signal_payload.get("updated_at")
                ),
                "catchup": True,
                "source_signal_time": recent_signal.get("time"),
            }
        )
        return catchup_payload

    def _execute_side(
        self,
        side: str,
        signal_payload: Dict[str, Any],
        position: Optional[Dict[str, Any]] = None,
    ):
        if position is None:
            market_state = self._read_market_state("주문 전 시장 상태 조회 실패")
            if not market_state:
                return
            position, balance = market_state
            self._set_market_state(position=position, balance=balance)
        elif self._market_entry_block_reason():
            self._set_safety_status(signal_payload, self._market_entry_block_reason())
            return

        if position["side"] == side.lower():
            self._set_status(signal_payload, f"이미 {side} 포지션 보유")
            self._sync_protection_orders(position)
            return

        block_reason = self._entry_block_reason_for(position)
        if position["side"] == "flat" and block_reason:
            self._set_safety_status(signal_payload, block_reason)
            return

        close_orders = []
        if position["side"] != "flat":
            self._cancel_protection_orders("CANCEL_PROTECTION")

        if position["side"] == "long" and side == "SHORT":
            order = self.exchange.close_long(self.symbol, position["amount"])
            if not order:
                self._set_error("CLOSE_LONG 주문 실패")
                return
            close_orders.append(self._order_record("CLOSE_LONG", order, signal_payload))
            self._record_closed_position(position, self._exit_price(position, signal_payload))
        elif position["side"] == "short" and side == "LONG":
            order = self.exchange.close_short(self.symbol, position["amount"])
            if not order:
                self._set_error("CLOSE_SHORT 주문 실패")
                return
            close_orders.append(self._order_record("CLOSE_SHORT", order, signal_payload))
            self._record_closed_position(position, self._exit_price(position, signal_payload))

        if close_orders:
            with self._lock:
                self.order_history.extend(close_orders)
                self._last_action = close_orders[-1]["action"]
                self._last_error = ""
            market_state = self._read_market_state("전환 후 시장 상태 조회 실패")
            if not market_state:
                return
            position, balance = market_state
            self._set_market_state(position=position, balance=balance)
            block_reason = self._entry_block_reason_for(position)
            if block_reason:
                self._set_safety_status(signal_payload, block_reason)
                return

        amount = self.exchange.amount_from_usdt(self.symbol, LONG_SHORT_ORDER_USDT)
        if amount <= 0:
            self._set_error("주문 수량 계산 실패")
            return

        if side == "LONG":
            order = self.exchange.open_long(self.symbol, amount)
            action = "OPEN_LONG"
        else:
            order = self.exchange.open_short(self.symbol, amount)
            action = "OPEN_SHORT"

        if order:
            record = self._order_record(action, order, signal_payload)
            with self._lock:
                self.order_history.append(record)
                self._daily_trade_count += 1
                self._last_action = action
                self._last_error = ""
            market_state = self._read_market_state("주문 후 시장 상태 조회 실패")
            if market_state:
                position, balance = market_state
                self._set_market_state(position=position, balance=balance)
            self._ensure_protection_orders(position)
        else:
            self._set_error(f"{action} 주문 실패")

    def _exit_for_safety(
        self, position: Dict[str, Any], signal_payload: Dict[str, Any]
    ) -> bool:
        reason = self._daily_loss_limit_reason(position)
        if not reason:
            return False

        if position["side"] == "flat":
            self._set_safety_status(signal_payload, reason)
            return False

        current_price = self._exit_price(position, signal_payload)
        if position["side"] == "long":
            self._cancel_protection_orders("CANCEL_PROTECTION")
            order = self.exchange.close_long(self.symbol, position["amount"])
            action = "DAILY_STOP_LONG"
            side = "LONG"
        else:
            self._cancel_protection_orders("CANCEL_PROTECTION")
            order = self.exchange.close_short(self.symbol, position["amount"])
            action = "DAILY_STOP_SHORT"
            side = "SHORT"

        if not order:
            self._set_error(f"{action} 주문 실패")
            return True

        self._record_closed_position(position, current_price)
        safety_payload = dict(signal_payload)
        safety_payload.update({
            "side": side,
            "reason": reason,
            "price": current_price,
        })
        with self._lock:
            self.order_history.append(
                self._order_record(action, order, safety_payload)
            )
            self._last_action = action
            self._last_error = ""
            self._last_position = self._flat_position()
            self._last_protection_orders = []
            self._last_safety_reason = reason
        return True

    def _exit_for_risk(
        self, position: Dict[str, Any], signal_payload: Dict[str, Any]
    ) -> bool:
        if position["side"] == "flat":
            return False

        entry_price = float(position.get("entry_price") or 0.0)
        current_price = float(
            position.get("mark_price")
            or signal_payload.get("price")
            or 0.0
        )
        if entry_price <= 0 or current_price <= 0:
            return False

        stop_loss, take_profit = self._risk_levels(position["side"], entry_price)
        action = ""
        reason = ""

        if position["side"] == "long":
            if current_price <= stop_loss:
                action = "STOP_LONG"
                reason = f"롱 손절: {current_price:.2f} <= {stop_loss:.2f}"
            elif current_price >= take_profit:
                action = "TAKE_LONG"
                reason = f"롱 익절: {current_price:.2f} >= {take_profit:.2f}"
        elif position["side"] == "short":
            if current_price >= stop_loss:
                action = "STOP_SHORT"
                reason = f"숏 손절: {current_price:.2f} >= {stop_loss:.2f}"
            elif current_price <= take_profit:
                action = "TAKE_SHORT"
                reason = f"숏 익절: {current_price:.2f} <= {take_profit:.2f}"

        if not action:
            return False

        if position["side"] == "long":
            self._cancel_protection_orders("CANCEL_PROTECTION")
            order = self.exchange.close_long(self.symbol, position["amount"])
            side = "LONG"
        else:
            self._cancel_protection_orders("CANCEL_PROTECTION")
            order = self.exchange.close_short(self.symbol, position["amount"])
            side = "SHORT"

        if not order:
            self._set_error(f"{action} 주문 실패")
            return True

        self._record_closed_position(position, current_price)
        risk_payload = dict(signal_payload)
        risk_payload.update({
            "side": side,
            "reason": reason,
            "price": current_price,
        })
        with self._lock:
            self.order_history.append(self._order_record(action, order, risk_payload))
            self._last_action = action
            self._last_error = ""
            self._last_position = self._flat_position()
            self._last_protection_orders = []
        return True

    def _entry_block_reason_for(self, position: Dict[str, Any]) -> str:
        return (
            self._market_entry_block_reason()
            or self._safety_status(position).get("block_reason", "")
        )

    def _daily_loss_limit_reason(self, position: Dict[str, Any]) -> str:
        reason = self._safety_status(position).get("block_reason", "")
        if reason.startswith("일일 손실"):
            return reason
        return ""

    def _safety_status(
        self,
        position: Optional[Dict[str, Any]] = None,
        balance: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            safety_day = self._safety_day
            day_start_balance = self._day_start_balance
            daily_trade_count = self._daily_trade_count
            consecutive_losses = self._consecutive_losses
            cooldown_until = self._cooldown_until
            last_reason = self._last_safety_reason
            if position is None:
                position = dict(self._last_position)
            else:
                position = dict(position)
            if balance is None:
                balance = self._last_balance

        daily_pnl = self._daily_pnl(float(balance or 0.0), position, day_start_balance)
        daily_loss_usdt = max(0.0, -daily_pnl)
        daily_loss_pct = 0.0
        if day_start_balance and day_start_balance > 0:
            daily_loss_pct = daily_loss_usdt / day_start_balance * 100

        block_reason = self._block_reason(
            daily_loss_usdt,
            daily_loss_pct,
            daily_trade_count,
            cooldown_until,
        )
        return {
            "day": safety_day.isoformat(),
            "day_start_balance": day_start_balance or 0.0,
            "daily_pnl": daily_pnl,
            "daily_loss_usdt": daily_loss_usdt,
            "daily_loss_pct": daily_loss_pct,
            "max_daily_loss_pct": LONG_SHORT_MAX_DAILY_LOSS_PCT,
            "max_daily_loss_usdt": LONG_SHORT_MAX_DAILY_LOSS_USDT,
            "daily_trade_count": daily_trade_count,
            "max_daily_trades": LONG_SHORT_MAX_DAILY_TRADES,
            "consecutive_losses": consecutive_losses,
            "max_consecutive_losses": LONG_SHORT_MAX_CONSECUTIVE_LOSSES,
            "cooldown_after_losses_minutes": LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES,
            "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
            "blocked": bool(block_reason),
            "block_reason": block_reason,
            "last_reason": last_reason,
        }

    def _block_reason(
        self,
        daily_loss_usdt: float,
        daily_loss_pct: float,
        daily_trade_count: int,
        cooldown_until: Optional[datetime],
    ) -> str:
        if (
            LONG_SHORT_MAX_DAILY_LOSS_USDT > 0
            and daily_loss_usdt >= LONG_SHORT_MAX_DAILY_LOSS_USDT
        ):
            return (
                f"일일 손실 한도 도달: "
                f"-${daily_loss_usdt:.2f} / ${LONG_SHORT_MAX_DAILY_LOSS_USDT:.2f}"
            )
        if (
            LONG_SHORT_MAX_DAILY_LOSS_PCT > 0
            and daily_loss_pct >= LONG_SHORT_MAX_DAILY_LOSS_PCT
        ):
            return (
                f"일일 손실 한도 도달: "
                f"-{daily_loss_pct:.2f}% / {LONG_SHORT_MAX_DAILY_LOSS_PCT:.2f}%"
            )
        if cooldown_until and datetime.now() < cooldown_until:
            remaining = max(1, int((cooldown_until - datetime.now()).total_seconds() // 60))
            return f"연속 손실 쿨다운: 약 {remaining}분 남음"
        if (
            LONG_SHORT_MAX_DAILY_TRADES > 0
            and daily_trade_count >= LONG_SHORT_MAX_DAILY_TRADES
        ):
            return (
                f"일일 최대 거래 횟수 도달: "
                f"{daily_trade_count}/{LONG_SHORT_MAX_DAILY_TRADES}"
            )
        return ""

    @staticmethod
    def _daily_pnl(
        balance: float,
        position: Dict[str, Any],
        day_start_balance: Optional[float],
    ) -> float:
        if not day_start_balance or day_start_balance <= 0:
            return 0.0
        unrealized_pnl = float(position.get("unrealized_pnl") or 0.0)
        return balance - day_start_balance + unrealized_pnl

    def _cached_market_state(self) -> tuple[Dict[str, Any], float]:
        with self._lock:
            return dict(self._last_position), self._last_balance

    def _record_external_position_close(
        self,
        previous_position: Dict[str, Any],
        previous_balance: float,
        current_position: Dict[str, Any],
        current_balance: float,
    ):
        if previous_position["side"] == "flat" or current_position["side"] != "flat":
            return

        realized_pnl = current_balance - previous_balance
        if abs(realized_pnl) < 1e-8:
            realized_pnl = float(previous_position.get("unrealized_pnl") or 0.0)
        self._record_closed_position(
            previous_position,
            float(previous_position.get("mark_price") or 0.0),
            realized_pnl,
        )

    def _record_closed_position(
        self,
        position: Dict[str, Any],
        exit_price: float,
        realized_pnl: Optional[float] = None,
    ):
        pnl = realized_pnl
        if pnl is None:
            pnl = self._position_pnl(position, exit_price)

        with self._lock:
            if pnl < 0:
                self._consecutive_losses += 1
            elif pnl > 0:
                self._consecutive_losses = 0

            if (
                LONG_SHORT_MAX_CONSECUTIVE_LOSSES > 0
                and LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES > 0
                and self._consecutive_losses >= LONG_SHORT_MAX_CONSECUTIVE_LOSSES
            ):
                self._cooldown_until = datetime.now() + timedelta(
                    minutes=LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES
                )
                self._last_safety_reason = (
                    f"연속 손실 {self._consecutive_losses}회, "
                    f"{LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES}분 쿨다운"
                )

    def _exit_price(
        self,
        position: Dict[str, Any],
        signal_payload: Dict[str, Any],
    ) -> float:
        return float(
            signal_payload.get("price")
            or position.get("mark_price")
            or position.get("entry_price")
            or 0.0
        )

    @staticmethod
    def _position_pnl(position: Dict[str, Any], exit_price: float) -> float:
        entry_price = float(position.get("entry_price") or 0.0)
        amount = float(position.get("amount") or 0.0)
        if entry_price <= 0 or exit_price <= 0 or amount <= 0:
            return 0.0
        if position["side"] == "short":
            return (entry_price - exit_price) * amount
        return (exit_price - entry_price) * amount

    def _sync_protection_orders(self, position: Dict[str, Any]):
        try:
            open_orders = self.exchange.fetch_protection_orders(self.symbol)
        except ExchangeUnavailableError as e:
            self._set_market_error(e, "보호 주문 조회 실패")
            return
        self._set_protection_orders(open_orders)
        with self._lock:
            self._protection_reconciled = True

        if position["side"] == "flat":
            if open_orders:
                self._cancel_protection_orders("CANCEL_STALE_PROTECTION")
            return

        if len(open_orders) < 2:
            if open_orders:
                self._cancel_protection_orders("CANCEL_INCOMPLETE_PROTECTION")
            self._ensure_protection_orders(position)

    def _ensure_protection_orders(self, position: Dict[str, Any]) -> List[Dict[str, Any]]:
        if position["side"] == "flat":
            return []

        entry_price = float(position.get("entry_price") or 0.0)
        amount = float(position.get("amount") or 0.0)
        if entry_price <= 0 or amount <= 0:
            self._set_error("보호 주문 생성 실패: 포지션 진입가/수량 없음")
            return []

        stop_loss, take_profit = self._risk_levels(position["side"], entry_price)
        created = self.exchange.create_protection_orders(
            self.symbol,
            position["side"],
            amount,
            stop_loss,
            take_profit,
        )
        self._set_protection_orders(created)

        records = []
        for order in created:
            action = self._protection_action(order)
            price = self._protection_stop_price(order)
            records.append(
                self._order_record(
                    action,
                    order,
                    {
                        "side": position["side"].upper(),
                        "price": price,
                        "confidence": 0.0,
                        "reason": "거래소 보호 주문 설정",
                    },
                )
            )

        with self._lock:
            self.order_history.extend(records)
            if len(created) >= 2:
                self._last_action = "보호 주문 설정"
                self._last_error = ""
            else:
                self._last_error = "보호 주문 생성 미완료"
        return created

    def _cancel_protection_orders(self, action: str) -> List[Dict[str, Any]]:
        try:
            cancelled = self.exchange.cancel_protection_orders(self.symbol)
        except ExchangeUnavailableError as e:
            self._set_market_error(e, "보호 주문 취소 전 조회 실패")
            return []
        self._set_protection_orders([])

        records = [
            self._order_record(
                action,
                order,
                {
                    "side": "HOLD",
                    "price": self._protection_stop_price(order),
                    "confidence": 0.0,
                    "reason": "거래소 보호 주문 취소",
                },
            )
            for order in cancelled
        ]
        if records:
            with self._lock:
                self.order_history.extend(records)
                self._last_action = action
        return cancelled

    def _risk_signal_payload(self, position: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            payload = dict(self._last_signal)
        payload.update({
            "side": position["side"].upper() if position["side"] != "flat" else "HOLD",
            "price": float(position.get("mark_price") or payload.get("price") or 0.0),
            "confidence": payload.get("confidence", 0.0),
            "reason": payload.get("reason", "리스크 감시"),
        })
        return payload

    @staticmethod
    def _protection_action(order: Dict[str, Any]) -> str:
        info = order.get("info", {})
        order_type = str(
            order.get("type") or info.get("type") or info.get("origType") or ""
        ).upper()
        if "TAKE_PROFIT" in order_type:
            return "SET_TAKE_PROFIT"
        return "SET_STOP_LOSS"

    @staticmethod
    def _protection_stop_price(order: Dict[str, Any]) -> float:
        info = order.get("info", {})
        for source in (order, info):
            for key in ("stopPrice", "triggerPrice", "price"):
                value = source.get(key)
                if value not in (None, ""):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        return 0.0

    def _build_signal_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        signal = self.strategy.analyze(df, self.symbol)
        indicators = self.strategy.get_indicators(df)
        latest = indicators.iloc[-1]

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": signal.signal.value,
            "side": self._side(signal.signal),
            "bias": self._bias(latest),
            "confidence": self._number(signal.confidence),
            "price": self._number(signal.price),
            "reason": signal.reason,
            "updated_at": df.index[-1].isoformat(),
            "ema_fast": self._number(latest.get("ema_fast")),
            "ema_slow": self._number(latest.get("ema_slow")),
            "ema_slope": self._number(latest.get("ema_slope")),
            "rsi": self._number(latest.get("rsi")),
            "trend_gap": self._number(latest.get("trend_gap")),
            "recent_signals": self._recent_signals(df),
        }

    def _recent_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        config = self.strategy.config
        min_rows = max(config.slow_ema, config.rsi_period) + config.slope_period + 2
        start = max(min_rows, len(df) - 120)

        for i in range(start, len(df)):
            window = df.iloc[: i + 1]
            signal = self.strategy.analyze(window, self.symbol)
            if signal.signal == Signal.HOLD:
                continue
            rows.append(
                {
                    "time": window.index[-1].isoformat(),
                    "signal": signal.signal.value,
                    "side": self._side(signal.signal),
                    "price": self._number(signal.price),
                    "confidence": self._number(signal.confidence),
                    "reason": signal.reason,
                }
            )

        return rows[-8:]

    def _positions_payload(self, position: Dict[str, Any]) -> List[Dict[str, Any]]:
        if position["side"] == "flat":
            return []

        entry_price = float(position.get("entry_price") or 0.0)
        stop_loss, take_profit = self._risk_levels(position["side"], entry_price)
        return [
            {
                "symbol": self.symbol,
                "side": position["side"],
                "entry_price": entry_price,
                "mark_price": float(position.get("mark_price") or 0.0),
                "amount": position["amount"],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "unrealized_pnl": float(position.get("unrealized_pnl") or 0.0),
                "liquidation_price": float(position.get("liquidation_price") or 0.0),
                "pnl_pct": float(position.get("percentage") or 0.0),
                "strategy": f"Futures {position['side'].upper()}",
            }
        ]

    def _risk_levels(self, side: str, entry_price: float) -> tuple[float, float]:
        if entry_price <= 0:
            return 0.0, 0.0
        if side == "short":
            stop_loss = entry_price * (1 + LONG_SHORT_STOP_LOSS_PCT / 100)
            take_profit = entry_price * (1 - LONG_SHORT_TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = entry_price * (1 - LONG_SHORT_STOP_LOSS_PCT / 100)
            take_profit = entry_price * (1 + LONG_SHORT_TAKE_PROFIT_PCT / 100)
        return stop_loss, take_profit

    def _order_record(
        self, action: str, order: Dict[str, Any], signal_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "time": datetime.now().isoformat(),
            "action": action,
            "order_id": order.get("id"),
            "symbol": self.symbol,
            "side": signal_payload.get("side", "HOLD"),
            "price": signal_payload.get("price", 0.0),
            "confidence": signal_payload.get("confidence", 0.0),
            "reason": signal_payload.get("reason", "-"),
        }

    def _set_status(self, signal_payload: Dict[str, Any], action: str):
        with self._lock:
            self._last_signal = signal_payload
            self._last_action = action

    def _set_safety_status(self, signal_payload: Dict[str, Any], reason: str):
        with self._lock:
            self._last_signal = signal_payload
            self._last_action = f"안전장치: {reason}"
            self._last_safety_reason = reason

    def _set_error(self, error: str):
        with self._lock:
            self._last_error = mask_sensitive(error)

    def _read_market_state(
        self, context: str
    ) -> Optional[tuple[Dict[str, Any], float]]:
        try:
            position = self.exchange.fetch_position(self.symbol)
            balance = self.exchange.get_balance("USDT")
            return position, balance
        except ExchangeUnavailableError as e:
            self._set_market_error(e, context)
            return None

    def _set_market_error(self, error: ExchangeUnavailableError, context: str):
        status = "rate_limited" if error.rate_limited else "error"
        message = mask_sensitive(error)
        block_reason = (
            "거래소 rate limit으로 신규 진입 차단"
            if status == "rate_limited"
            else f"{context}: {message}"
        )
        with self._lock:
            self._market_state_status = status
            self._last_market_error = message
            self._entry_block_reason = block_reason
            self._last_error = message

    def _set_market_stale(self, reason: str):
        with self._lock:
            self._market_state_status = "stale"
            self._entry_block_reason = reason
            self._last_market_error = reason

    def _market_entry_block_reason(self) -> str:
        with self._lock:
            if self._market_state_status == "ok":
                return ""
            if self._market_state_status == "rate_limited":
                return "거래소 rate limit으로 신규 진입 차단"
            return self._entry_block_reason or "시장 상태 미확인"

    def _set_market_state(
        self,
        position: Optional[Dict[str, Any]] = None,
        balance: Optional[float] = None,
    ):
        with self._lock:
            if position is not None:
                self._last_position = dict(position)
            if balance is not None:
                self._last_balance = float(balance or 0.0)
                self._sync_safety_day_locked(self._last_balance)
            if position is not None or balance is not None:
                self._market_state_status = "ok"
                self._entry_block_reason = ""
                self._last_market_error = ""
                self._last_market_update_at = datetime.now().isoformat()

    def _set_protection_orders(self, orders: List[Dict[str, Any]]):
        with self._lock:
            self._last_protection_orders = list(orders)

    def _should_skip_flat_protection_sync(self, position: Dict[str, Any]) -> bool:
        if position["side"] != "flat":
            return False
        with self._lock:
            return self._protection_reconciled and not self._last_protection_orders

    def _sync_safety_day_locked(self, balance: float):
        today = datetime.now().date()
        if self._safety_day != today:
            self._safety_day = today
            self._day_start_balance = balance
            self._daily_trade_count = 0
            self._consecutive_losses = 0
            self._cooldown_until = None
            self._last_safety_reason = ""
            return

        if self._day_start_balance is None:
            self._day_start_balance = balance

        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            self._cooldown_until = None
            if self._last_safety_reason.startswith("연속 손실"):
                self._last_safety_reason = ""

    def _is_stale(self, updated_at: str) -> bool:
        try:
            signal_time = datetime.fromisoformat(updated_at)
        except ValueError:
            return True
        age = (datetime.utcnow() - signal_time).total_seconds() / 60
        return age > LONG_SHORT_MAX_SIGNAL_AGE_MINUTES

    @staticmethod
    def _latest_recent_signal(signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not signals:
            return None
        return max(signals, key=lambda signal: signal.get("time", ""))

    @staticmethod
    def _bias_side(bias: Optional[str]) -> str:
        if bias == "LONG_BIAS":
            return "LONG"
        if bias == "SHORT_BIAS":
            return "SHORT"
        return "HOLD"

    def _closed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        timeframe_seconds = self._timeframe_seconds()
        if timeframe_seconds <= 0:
            return df

        latest = df.index[-1]
        if getattr(latest, "tzinfo", None):
            latest = latest.tz_convert(None)
        close_time = latest.to_pydatetime() + timedelta(seconds=timeframe_seconds)
        if datetime.utcnow() < close_time:
            return df.iloc[:-1]
        return df

    def _timeframe_seconds(self) -> int:
        unit = self.timeframe[-1]
        try:
            value = int(self.timeframe[:-1])
        except ValueError:
            return 0
        if unit == "m":
            return value * 60
        if unit == "h":
            return value * 60 * 60
        if unit == "d":
            return value * 24 * 60 * 60
        return 0

    @staticmethod
    def _flat_position() -> Dict[str, Any]:
        return {
            "side": "flat",
            "amount": 0.0,
            "entry_price": 0.0,
            "mark_price": 0.0,
            "unrealized_pnl": 0.0,
            "liquidation_price": 0.0,
            "percentage": 0.0,
        }

    @staticmethod
    def _bias(row: pd.Series) -> str:
        if row["ema_fast"] > row["ema_slow"] and row["ema_slope"] > 0:
            return "LONG_BIAS"
        if row["ema_fast"] < row["ema_slow"] and row["ema_slope"] < 0:
            return "SHORT_BIAS"
        return "NEUTRAL"

    @staticmethod
    def _side(signal: Signal) -> str:
        if signal == Signal.BUY:
            return "LONG"
        if signal == Signal.SELL:
            return "SHORT"
        return "HOLD"

    def _empty_signal_payload(self, reason: str) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": Signal.HOLD.value,
            "side": "HOLD",
            "bias": "NEUTRAL",
            "confidence": 0.0,
            "price": 0.0,
            "reason": reason,
            "updated_at": datetime.utcnow().isoformat(),
            "recent_signals": [],
        }

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
