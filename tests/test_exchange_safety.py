import unittest
from unittest.mock import patch

import config
import pandas as pd

from exchange import BinanceExchange, _maybe_resample_ohlcv, _ohlcv_frame, _source_ohlcv_request


class FakeCcxtExchange:
    def __init__(self):
        self.order_calls = 0
        self.cancel_calls = 0

    def fetch_ticker(self, symbol):
        return {"symbol": symbol, "last": 123.45}

    def create_market_buy_order(self, symbol, amount):
        self.order_calls += 1
        raise AssertionError("real buy order should not be called in dry-run")

    def create_market_sell_order(self, symbol, amount):
        self.order_calls += 1
        raise AssertionError("real sell order should not be called in dry-run")

    def create_limit_buy_order(self, symbol, amount, price):
        self.order_calls += 1
        raise AssertionError("real limit buy order should not be called in dry-run")

    def create_limit_sell_order(self, symbol, amount, price):
        self.order_calls += 1
        raise AssertionError("real limit sell order should not be called in dry-run")

    def cancel_order(self, order_id, symbol):
        self.cancel_calls += 1
        raise AssertionError("real cancel should not be called in dry-run")

    def fetch_order(self, order_id, symbol):
        raise AssertionError("real order status should not be called in dry-run")


def make_dry_run_exchange():
    exchange = object.__new__(BinanceExchange)
    exchange.exchange = FakeCcxtExchange()
    exchange.dry_run = True
    return exchange


class ExchangeSafetyTest(unittest.TestCase):
    def test_dry_run_market_buy_returns_fake_order_without_real_order(self):
        exchange = make_dry_run_exchange()

        order = exchange.create_market_buy("BTC/USDT", 0.01, price=50000.0)

        self.assertEqual(order["symbol"], "BTC/USDT")
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["type"], "market")
        self.assertEqual(order["average"], 50000.0)
        self.assertEqual(order["filled"], 0.01)
        self.assertEqual(order["cost"], 500.0)
        self.assertTrue(order["id"].startswith("dry-run-buy-"))
        self.assertTrue(order["info"]["dry_run"])
        self.assertEqual(exchange.exchange.order_calls, 0)

    def test_dry_run_market_sell_uses_ticker_price_without_real_order(self):
        exchange = make_dry_run_exchange()

        order = exchange.create_market_sell("ETH/USDT", 2.0)

        self.assertEqual(order["side"], "sell")
        self.assertEqual(order["average"], 123.45)
        self.assertEqual(order["cost"], 246.9)
        self.assertEqual(exchange.exchange.order_calls, 0)

    def test_dry_run_limit_and_cancel_do_not_call_real_exchange(self):
        exchange = make_dry_run_exchange()

        buy = exchange.create_limit_buy("BTC/USDT", 0.01, 49000.0)
        sell = exchange.create_limit_sell("BTC/USDT", 0.01, 51000.0)
        cancelled = exchange.cancel_order("dry-run-buy-1", "BTC/USDT")

        self.assertEqual(buy["type"], "limit")
        self.assertEqual(sell["type"], "limit")
        self.assertTrue(cancelled)
        self.assertEqual(exchange.exchange.order_calls, 0)
        self.assertEqual(exchange.exchange.cancel_calls, 0)

    def test_dry_run_balance_uses_paper_balance(self):
        exchange = make_dry_run_exchange()

        self.assertEqual(exchange.get_balance("USDT"), config.DRY_RUN_STARTING_BALANCE)
        self.assertEqual(exchange.get_balance("BTC"), 0.0)
        self.assertEqual(
            exchange.get_all_balances(), {"USDT": config.DRY_RUN_STARTING_BALANCE}
        )

    def test_dry_run_order_status_does_not_call_real_exchange(self):
        exchange = make_dry_run_exchange()

        self.assertIsNone(exchange.get_order_status("dry-run-buy-1", "BTC/USDT"))

    def test_live_trading_requires_explicit_allow_flag(self):
        with patch("config.USE_TESTNET", False), patch("config.DRY_RUN", False), patch(
            "config.ALLOW_LIVE_TRADING", False
        ), patch("exchange.ccxt.binance") as binance:
            with self.assertRaises(RuntimeError):
                BinanceExchange()

            binance.assert_not_called()

    def test_mask_sensitive_redacts_configured_secrets(self):
        with patch("config.BINANCE_API_KEY", "abc123"), patch(
            "config.BINANCE_API_SECRET", "secret456"
        ):
            text = config.mask_sensitive("abc123 and secret456")

        self.assertNotIn("abc123", text)
        self.assertNotIn("secret456", text)
        self.assertEqual(text, "<masked> and <masked>")

    def test_mask_sensitive_redacts_signed_query_fields(self):
        text = config.mask_sensitive(
            "signature=abc&timestamp=123&recvWindow=10000 apiKey=key secret=sec token=t"
        )

        for value in ("abc", "123", "10000", "key", "sec", "t"):
            self.assertNotIn(value, text)
        for key in ("signature=", "timestamp=", "recvWindow=", "apiKey=", "secret=", "token="):
            self.assertNotIn(key, text)
        self.assertIn("<masked>=<masked>", text)

    def test_ten_minute_ohlcv_is_resampled_from_five_minute_rows(self):
        source_timeframe, source_limit, target_rule = _source_ohlcv_request("10m", 2)
        base = pd.Timestamp("2026-01-01 00:00:00")
        raw = []
        for i in range(4):
            timestamp_ms = int((base + pd.Timedelta(minutes=i * 5)).timestamp() * 1000)
            raw.append([timestamp_ms, i + 1, i + 3, i, i + 2, 10])

        df = _ohlcv_frame(raw)
        resampled = _maybe_resample_ohlcv(df, target_rule, 2)

        self.assertEqual(source_timeframe, "5m")
        self.assertEqual(source_limit, 6)
        self.assertEqual(len(resampled), 2)
        self.assertEqual(list(resampled["open"]), [1, 3])
        self.assertEqual(list(resampled["high"]), [4, 6])
        self.assertEqual(list(resampled["low"]), [0, 2])
        self.assertEqual(list(resampled["close"]), [3, 5])
        self.assertEqual(list(resampled["volume"]), [20, 20])


if __name__ == "__main__":
    unittest.main()
