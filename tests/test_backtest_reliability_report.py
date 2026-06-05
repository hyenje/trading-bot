import unittest
from datetime import datetime, timedelta

import pandas as pd

from backtesting.engine import BacktestResult, Trade
from main import (
    _equal_time_windows,
    _format_trade_distribution,
    _format_window_result_table,
)


def make_df(rows=120):
    index = pd.date_range("2026-01-01", periods=rows, freq="10min")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )


def make_trade(pnl, minutes=10):
    entry = datetime(2026, 1, 1)
    return Trade(
        entry_time=entry,
        exit_time=entry + timedelta(minutes=minutes),
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        exit_price=101.0,
        notional=25.0,
        pnl=pnl,
    )


class BacktestReliabilityReportTest(unittest.TestCase):
    def test_equal_time_windows_respects_minimum_rows(self):
        windows = _equal_time_windows(make_df(), 3, min_rows=30)

        self.assertEqual(len(windows), 3)
        self.assertTrue(all(len(window) >= 30 for window in windows))

    def test_window_result_table_includes_holdout_metrics(self):
        result = BacktestResult(
            total_trades=1,
            win_rate=100.0,
            total_pnl=1.5,
            max_drawdown=0.2,
            profit_factor=2.0,
        )

        text = _format_window_result_table("기간별 OOS 결과", [("OOS 1/3", make_df(), result)])

        self.assertIn("b&h%", text)
        self.assertIn("OOS 1/3", text)
        self.assertIn("1.50", text)

    def test_trade_distribution_reports_loss_streak(self):
        result = BacktestResult(
            trades=[make_trade(-1.0), make_trade(-0.5), make_trade(2.0)],
            best_trade=2.0,
            worst_trade=-1.0,
        )

        text = _format_trade_distribution(result)

        self.assertIn("max_consecutive_losses=2", text)
        self.assertIn("best_trade_profit_share=100.0%", text)


if __name__ == "__main__":
    unittest.main()
