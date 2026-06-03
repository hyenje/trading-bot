import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from config import LONG_SHORT_MAX_CONSECUTIVE_LOSSES, LONG_SHORT_MAX_DAILY_TRADES
from exchange import ExchangeUnavailableError
from long_short_executor import BTCLongShortExecutor


class FakeFuturesExchange:
    def __init__(
        self,
        side="flat",
        entry_price=100.0,
        mark_price=100.0,
        unrealized_pnl=0.0,
        balance=5000.0,
    ):
        self.balance = balance
        self.position = {
            "side": side,
            "amount": 0.01 if side != "flat" else 0.0,
            "entry_price": entry_price if side != "flat" else 0.0,
            "mark_price": mark_price if side != "flat" else 0.0,
            "unrealized_pnl": unrealized_pnl,
            "liquidation_price": 0.0,
            "percentage": 0.0,
        }
        self.actions = []
        self.protection_orders = []
        self.fetch_position_calls = 0
        self.get_balance_calls = 0
        self.fetch_protection_calls = 0
        self.position_error = None
        self.balance_error = None
        self.protection_error = None

    def fetch_position(self, symbol):
        self.fetch_position_calls += 1
        if self.position_error:
            raise self.position_error
        return self.position

    def amount_from_usdt(self, symbol, usdt_amount):
        return 0.001

    def open_long(self, symbol, amount):
        self.actions.append(("open_long", amount))
        self.position = {
            "side": "long",
            "amount": amount,
            "entry_price": 100.0,
            "mark_price": 100.0,
            "unrealized_pnl": 0.0,
            "liquidation_price": 0.0,
            "percentage": 0.0,
        }
        return {"id": "open-long-1"}

    def open_short(self, symbol, amount):
        self.actions.append(("open_short", amount))
        self.position = {
            "side": "short",
            "amount": amount,
            "entry_price": 100.0,
            "mark_price": 100.0,
            "unrealized_pnl": 0.0,
            "liquidation_price": 0.0,
            "percentage": 0.0,
        }
        return {"id": "open-short-1"}

    def close_long(self, symbol, amount):
        self.actions.append(("close_long", amount))
        self.position = {"side": "flat", "amount": 0.0}
        return {"id": "close-long-1"}

    def close_short(self, symbol, amount):
        self.actions.append(("close_short", amount))
        self.position = {"side": "flat", "amount": 0.0}
        return {"id": "close-short-1"}

    def get_balance(self, currency):
        self.get_balance_calls += 1
        if self.balance_error:
            raise self.balance_error
        return self.balance

    def create_protection_orders(self, symbol, position_side, amount, stop_loss, take_profit):
        self.actions.append(("create_protection", position_side, amount, stop_loss, take_profit))
        side = "sell" if position_side == "long" else "buy"
        self.protection_orders = [
            {
                "id": "protect-stop-1",
                "type": "STOP_MARKET",
                "side": side,
                "stopPrice": stop_loss,
                "clientOrderId": "btcls-sl-test",
            },
            {
                "id": "protect-take-1",
                "type": "TAKE_PROFIT_MARKET",
                "side": side,
                "stopPrice": take_profit,
                "clientOrderId": "btcls-tp-test",
            },
        ]
        return self.protection_orders

    def fetch_protection_orders(self, symbol):
        self.fetch_protection_calls += 1
        if self.protection_error:
            raise self.protection_error
        return list(self.protection_orders)

    def cancel_protection_orders(self, symbol):
        cancelled = list(self.protection_orders)
        if cancelled:
            self.actions.append(("cancel_protection", len(cancelled)))
        self.protection_orders = []
        return cancelled


def make_executor(exchange):
    executor = object.__new__(BTCLongShortExecutor)
    executor.symbol = "BTC/USDT"
    executor.timeframe = "10m"
    executor.exchange = exchange
    executor.order_history = []
    executor._lock = MagicMock()
    executor._lock.__enter__.return_value = None
    executor._lock.__exit__.return_value = None
    executor._last_action = "대기"
    executor._last_error = ""
    executor._last_signal = executor._empty_signal_payload("test")
    executor._last_signal_key = None
    executor._last_position = exchange.fetch_position("BTC/USDT")
    executor._last_balance = 5000.0
    executor._last_protection_orders = []
    executor._market_state_status = "ok"
    executor._last_market_error = ""
    executor._last_market_update_at = datetime.now().isoformat()
    executor._entry_block_reason = ""
    executor._protection_reconciled = False
    executor._safety_day = datetime.now().date()
    executor._day_start_balance = 5000.0
    executor._daily_trade_count = 0
    executor._consecutive_losses = 0
    executor._cooldown_until = None
    executor._last_safety_reason = ""
    executor.running = True
    return executor


class LongShortExecutorTest(unittest.TestCase):
    def test_execute_long_opens_long_when_flat(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)

        executor._execute_side("LONG", {"side": "LONG", "price": 100, "confidence": 1, "reason": "test"})

        self.assertEqual(exchange.actions[0], ("open_long", 0.001))
        self.assertEqual(exchange.actions[1][0], "create_protection")
        self.assertEqual(executor._daily_trade_count, 1)
        self.assertEqual(executor.order_history[0]["action"], "OPEN_LONG")
        self.assertIn("SET_STOP_LOSS", [o["action"] for o in executor.order_history])
        self.assertIn("SET_TAKE_PROFIT", [o["action"] for o in executor.order_history])

    def test_execute_short_closes_long_then_opens_short(self):
        exchange = FakeFuturesExchange(side="long")
        executor = make_executor(exchange)

        executor._execute_side("SHORT", {"side": "SHORT", "price": 100, "confidence": 1, "reason": "test"})

        self.assertEqual(exchange.actions[0], ("close_long", 0.01))
        self.assertEqual(exchange.actions[1], ("open_short", 0.001))
        self.assertEqual(exchange.actions[2][0], "create_protection")
        self.assertEqual(
            [o["action"] for o in executor.order_history[:2]],
            ["CLOSE_LONG", "OPEN_SHORT"],
        )

    def test_risk_exit_closes_long_at_stop_loss(self):
        exchange = FakeFuturesExchange(side="long", entry_price=100.0, mark_price=98.0)
        executor = make_executor(exchange)

        exited = executor._exit_for_risk(
            exchange.fetch_position("BTC/USDT"),
            {"side": "HOLD", "price": 98.0, "confidence": 0, "reason": "test"},
        )

        self.assertTrue(exited)
        self.assertEqual(exchange.actions, [("close_long", 0.01)])
        self.assertEqual(executor.order_history[0]["action"], "STOP_LONG")

    def test_risk_exit_closes_short_at_take_profit(self):
        exchange = FakeFuturesExchange(side="short", entry_price=100.0, mark_price=96.0)
        executor = make_executor(exchange)

        exited = executor._exit_for_risk(
            exchange.fetch_position("BTC/USDT"),
            {"side": "HOLD", "price": 96.0, "confidence": 0, "reason": "test"},
        )

        self.assertTrue(exited)
        self.assertEqual(exchange.actions, [("close_short", 0.01)])
        self.assertEqual(executor.order_history[0]["action"], "TAKE_SHORT")

    def test_get_status_uses_cached_market_state(self):
        exchange = FakeFuturesExchange(side="long", entry_price=100.0, mark_price=101.0)
        executor = make_executor(exchange)
        exchange.fetch_position_calls = 0
        exchange.get_balance_calls = 0

        status = executor.get_status()

        self.assertFalse(status["observer_mode"])
        self.assertTrue(status["execution_mode"])
        self.assertEqual(status["balance"], 5000.0)
        self.assertEqual(status["open_positions"], 1)
        self.assertEqual(status["executor_status"]["market_state_status"], "ok")
        self.assertEqual(status["executor_status"]["entry_block_reason"], "")
        self.assertEqual(exchange.fetch_position_calls, 0)
        self.assertEqual(exchange.get_balance_calls, 0)

    def test_sync_protection_recreates_missing_orders_for_open_position(self):
        exchange = FakeFuturesExchange(side="long", entry_price=100.0, mark_price=101.0)
        executor = make_executor(exchange)

        executor._sync_protection_orders(exchange.fetch_position("BTC/USDT"))

        self.assertEqual(exchange.actions[0][0], "create_protection")
        self.assertEqual(len(executor._last_protection_orders), 2)

    def test_sync_protection_cancels_stale_orders_when_flat(self):
        exchange = FakeFuturesExchange()
        exchange.protection_orders = [
            {"id": "protect-stop-1", "type": "STOP_MARKET", "clientOrderId": "btcls-sl-test"}
        ]
        executor = make_executor(exchange)

        executor._sync_protection_orders(exchange.fetch_position("BTC/USDT"))

        self.assertEqual(exchange.actions, [("cancel_protection", 1)])
        self.assertEqual(executor._last_protection_orders, [])

    def test_risk_tick_skips_repeated_flat_protection_fetch_after_reconcile(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)
        executor._protection_reconciled = True
        exchange.fetch_protection_calls = 0

        executor._risk_tick()

        self.assertEqual(exchange.fetch_protection_calls, 0)

    def test_max_daily_trades_blocks_new_entry(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)
        executor._daily_trade_count = LONG_SHORT_MAX_DAILY_TRADES

        executor._execute_side("LONG", {"side": "LONG", "price": 100, "confidence": 1, "reason": "test"})

        self.assertEqual(exchange.actions, [])
        self.assertIn("일일 최대 거래 횟수", executor._last_action)

    def test_daily_loss_limit_blocks_flat_entry(self):
        exchange = FakeFuturesExchange(balance=4940.0)
        executor = make_executor(exchange)
        executor._last_balance = 4940.0

        executor._execute_side("LONG", {"side": "LONG", "price": 100, "confidence": 1, "reason": "test"})

        self.assertEqual(exchange.actions, [])
        self.assertIn("일일 손실 한도", executor._last_action)

    def test_rate_limit_blocks_entry_without_order(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)
        exchange.position_error = ExchangeUnavailableError(
            "binance 429 Too Many Requests",
            rate_limited=True,
        )

        executor._execute_side("LONG", {"side": "LONG", "price": 100, "confidence": 1, "reason": "test"})

        self.assertEqual(exchange.actions, [])
        self.assertEqual(executor._market_state_status, "rate_limited")
        self.assertIn("rate limit", executor._entry_block_reason)

    def test_position_fetch_failure_keeps_last_known_position(self):
        exchange = FakeFuturesExchange(side="long", entry_price=100.0, mark_price=101.0)
        executor = make_executor(exchange)
        exchange.position_error = ExchangeUnavailableError(
            "binance 418 I'm a teapot",
            rate_limited=True,
        )

        executor._risk_tick()

        self.assertEqual(executor._last_position["side"], "long")
        self.assertEqual(executor._last_position["entry_price"], 100.0)
        self.assertEqual(executor._market_state_status, "rate_limited")
        self.assertEqual(exchange.actions, [])

    def test_recent_signal_catchup_promotes_matching_fresh_signal_when_enabled(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)
        signal_time = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        payload = {
            "signal": "HOLD",
            "side": "HOLD",
            "bias": "LONG_BIAS",
            "confidence": 0.0,
            "price": 101.0,
            "reason": "전환 없음",
            "updated_at": datetime.utcnow().isoformat(),
            "recent_signals": [
                {
                    "time": signal_time,
                    "signal": "BUY",
                    "side": "LONG",
                    "price": 100.0,
                    "confidence": 0.8,
                    "reason": "recent long",
                }
            ],
        }

        with patch("long_short_executor.LONG_SHORT_ENABLE_SIGNAL_CATCHUP", True):
            catchup = executor._signal_payload_for_entry(
                payload,
                exchange.fetch_position("BTC/USDT"),
            )

        self.assertEqual(catchup["side"], "LONG")
        self.assertTrue(catchup["catchup"])
        self.assertEqual(catchup["updated_at"], signal_time)
        self.assertIn("최근 신호 따라잡기", catchup["reason"])

    def test_recent_signal_catchup_ignores_mismatched_bias(self):
        exchange = FakeFuturesExchange()
        executor = make_executor(exchange)
        payload = {
            "signal": "HOLD",
            "side": "HOLD",
            "bias": "SHORT_BIAS",
            "updated_at": datetime.utcnow().isoformat(),
            "recent_signals": [
                {
                    "time": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "signal": "BUY",
                    "side": "LONG",
                    "price": 100.0,
                    "confidence": 0.8,
                    "reason": "recent long",
                }
            ],
        }

        with patch("long_short_executor.LONG_SHORT_ENABLE_SIGNAL_CATCHUP", True):
            catchup = executor._signal_payload_for_entry(
                payload,
                exchange.fetch_position("BTC/USDT"),
            )

        self.assertIs(catchup, payload)

    def test_daily_loss_limit_closes_open_position(self):
        exchange = FakeFuturesExchange(side="long", unrealized_pnl=-60.0)
        executor = make_executor(exchange)

        exited = executor._exit_for_safety(
            exchange.fetch_position("BTC/USDT"),
            {"side": "LONG", "price": 98.0, "confidence": 0, "reason": "test"},
        )

        self.assertTrue(exited)
        self.assertEqual(exchange.actions, [("close_long", 0.01)])
        self.assertEqual(executor.order_history[0]["action"], "DAILY_STOP_LONG")

    def test_consecutive_losses_start_cooldown_and_block_entry(self):
        exchange = FakeFuturesExchange(side="long", entry_price=100.0, mark_price=98.0)
        executor = make_executor(exchange)
        executor._consecutive_losses = LONG_SHORT_MAX_CONSECUTIVE_LOSSES - 1

        exited = executor._exit_for_risk(
            exchange.fetch_position("BTC/USDT"),
            {"side": "HOLD", "price": 98.0, "confidence": 0, "reason": "test"},
        )
        executor._execute_side("LONG", {"side": "LONG", "price": 100, "confidence": 1, "reason": "test"})

        self.assertTrue(exited)
        self.assertIsNotNone(executor._cooldown_until)
        self.assertEqual(exchange.actions, [("close_long", 0.01)])
        self.assertIn("연속 손실 쿨다운", executor._last_action)

    def test_execution_guard_requires_explicit_flags(self):
        with patch("config.ENABLE_LONG_SHORT_EXECUTION", False):
            with self.assertRaises(RuntimeError):
                __import__("config").validate_long_short_execution_allowed()


if __name__ == "__main__":
    unittest.main()
