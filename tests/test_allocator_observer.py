import unittest

import numpy as np
import pandas as pd

from allocator_observer import AllocatorObserver, format_allocator_signal
from backtesting.market_regime_allocator import DEFENSIVE_ASSETS, RISK_ASSETS


def make_prices(rows=230):
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    steps = np.arange(rows)
    data = {}
    rates = {
        "SPY": 0.001,
        "QQQ": 0.002,
        "BTC": 0.003,
        "ETH": 0.002,
        "GLD": 0.0005,
        "TLT": -0.001,
        "SHY": 0.0001,
        "BIL": 0.0001,
        "VIX": 0.0,
    }
    for asset in RISK_ASSETS + DEFENSIVE_ASSETS:
        data[asset] = 100 * (1 + rates[asset]) ** steps
    data["VIX"] = np.full(rows, 18.0)
    return pd.DataFrame(data, index=index)


class AllocatorObserverTest(unittest.TestCase):
    def test_status_contains_allocator_signal(self):
        observer = AllocatorObserver(price_loader=lambda days: make_prices())

        status = observer.get_status()
        signal = status["allocator_signal"]

        self.assertTrue(status["allocator_observer_mode"])
        self.assertTrue(status["dry_run"])
        self.assertEqual(signal["strategy"], "tlt stress riskoff")
        self.assertIn("nonzero_allocation", signal)
        self.assertEqual(signal["error"], "")

    def test_format_allocator_signal_is_readable(self):
        observer = AllocatorObserver(price_loader=lambda days: make_prices())

        output = format_allocator_signal(observer.get_signal())

        self.assertIn("Allocator Signal", output)
        self.assertIn("tlt stress riskoff", output)
        self.assertIn("Target allocation", output)


if __name__ == "__main__":
    unittest.main()
