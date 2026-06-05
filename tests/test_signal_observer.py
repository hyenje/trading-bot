import unittest

import pandas as pd

from signal_observer import BTCSignalObserver
from strategies import BTCTrendLongShortStrategy
from config import BTCTrendLongShortConfig


class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, limit=240):
        freq = "4h" if timeframe == "4h" else "10min"
        index = pd.date_range("2026-01-01", periods=limit, freq=freq)
        close = pd.Series(range(100, 100 + limit), index=index, dtype=float)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
            }
        )

    def get_balance(self, currency):
        return 10000.0


def make_observer():
    observer = object.__new__(BTCSignalObserver)
    observer.symbol = "BTC/USDT"
    observer.timeframe = "10m"
    observer.exchange = FakeExchange()
    observer.strategy = BTCTrendLongShortStrategy(BTCTrendLongShortConfig())
    return observer


class BTCSignalObserverTest(unittest.TestCase):
    def test_status_contains_long_short_signal_payload(self):
        status = make_observer().get_status()
        signal = status["long_short_signal"]

        self.assertTrue(status["observer_mode"])
        self.assertEqual(status["balance"], 10000.0)
        self.assertEqual(signal["symbol"], "BTC/USDT")
        self.assertEqual(signal["timeframe"], "10m")
        self.assertIn(signal["side"], {"LONG", "SHORT", "HOLD"})
        self.assertIn(signal["bias"], {"LONG_BIAS", "SHORT_BIAS", "NEUTRAL"})
        self.assertIsInstance(signal["price"], float)
        self.assertIsInstance(signal["rsi"], float)
        self.assertEqual(signal["regime_timeframe"], "4h")
        self.assertIn(signal["regime_side"], {"LONG", "SHORT", "NEUTRAL"})
        self.assertIn("raw_side", signal)
        self.assertIn("entry_block_reason", signal)
        self.assertIn("reverse_block_reason", signal)
        self.assertIn("recent_signals", signal)


if __name__ == "__main__":
    unittest.main()
