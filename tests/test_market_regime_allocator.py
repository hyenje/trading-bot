import unittest

import numpy as np
import pandas as pd

from backtesting.market_regime_allocator import (
    ALL_ASSETS,
    AllocatorStrategyConfig,
    DEFENSIVE_ASSETS,
    RISK_ASSETS,
    build_candidate_benchmark_rows,
    build_extended_etf_rows,
    build_experiment_rows,
    build_rolling_window_rows,
    build_stress_period_rows,
    build_walk_forward_rows,
    run_allocator_backtest,
    select_allocation,
)


def trend_prices(rates, rows=230):
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    data = {}
    steps = np.arange(rows)
    for asset in RISK_ASSETS:
        data[asset] = 100 * (1 + rates.get(asset, 0.0)) ** steps
    return pd.DataFrame(data, index=index)


def trend_prices_with_defensive(rates, rows=230):
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    data = {}
    steps = np.arange(rows)
    for asset in RISK_ASSETS + DEFENSIVE_ASSETS:
        data[asset] = 100 * (1 + rates.get(asset, 0.0)) ** steps
    return pd.DataFrame(data, index=index)


def point_prices(points, base=100.0, rows=230):
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    data = {}
    for asset in RISK_ASSETS:
        values = np.full(rows, base, dtype=float)
        for offset, value in points.get(asset, {}).items():
            values[offset] = value
        data[asset] = values
    return pd.DataFrame(data, index=index)


class MarketRegimeAllocatorTest(unittest.TestCase):
    def test_risk_off_allocates_100_percent_cash(self):
        prices = trend_prices({"SPY": -0.002, "QQQ": 0.003, "BTC": 0.005, "ETH": 0.004})

        allocation, _, risk_on = select_allocation(prices, len(prices) - 1)

        self.assertFalse(risk_on)
        self.assertEqual(allocation["cash"], 1.0)
        self.assertEqual(sum(allocation[asset] for asset in RISK_ASSETS), 0.0)

    def test_risk_on_allocates_top_two_positive_scores_equally(self):
        prices = trend_prices({"SPY": 0.001, "QQQ": 0.003, "BTC": 0.005, "ETH": 0.002})

        allocation, scores, risk_on = select_allocation(prices, len(prices) - 1)

        self.assertTrue(risk_on)
        self.assertGreater(scores["BTC"], scores["QQQ"])
        self.assertEqual(allocation["BTC"], 0.5)
        self.assertEqual(allocation["QQQ"], 0.5)
        self.assertEqual(allocation["cash"], 0.0)

    def test_one_positive_score_allocates_60_percent_asset_40_percent_cash(self):
        prices = point_prices(
            {
                "SPY": {-181: 200, -91: 200, -1: 150},
                "QQQ": {-181: 180, -91: 180, -1: 100},
                "BTC": {-181: 100, -91: 100, -1: 150},
                "ETH": {-181: 180, -91: 180, -1: 100},
            },
            base=50,
        )

        allocation, scores, risk_on = select_allocation(prices, len(prices) - 1)

        self.assertTrue(risk_on)
        self.assertGreater(scores["BTC"], 0)
        self.assertLess(scores["SPY"], 0)
        self.assertEqual(allocation["BTC"], 0.6)
        self.assertEqual(allocation["cash"], 0.4)

    def test_all_negative_scores_allocates_100_percent_cash(self):
        prices = point_prices(
            {
                "SPY": {-181: 200, -91: 200, -1: 150},
                "QQQ": {-181: 180, -91: 180, -1: 100},
                "BTC": {-181: 180, -91: 180, -1: 100},
                "ETH": {-181: 180, -91: 180, -1: 100},
            },
            base=50,
        )

        allocation, scores, risk_on = select_allocation(prices, len(prices) - 1)

        self.assertTrue(risk_on)
        self.assertTrue(all(score < 0 for score in scores.values()))
        self.assertEqual(allocation["cash"], 1.0)

    def test_rebalance_cost_is_applied_to_risky_turnover_only(self):
        prices = trend_prices({"SPY": 0.002, "QQQ": 0.003, "BTC": 0.0, "ETH": 0.0})
        prices.iloc[-1] = prices.iloc[-2]

        result = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            fee_rate=0.001,
            start_date=prices.index[-1],
        )

        self.assertAlmostEqual(result.turnover, 1.0)
        self.assertAlmostEqual(result.total_cost, 10.0)
        self.assertAlmostEqual(result.final_equity, 9990.0)

    def test_rebalance_decision_does_not_use_current_day_price(self):
        prices = trend_prices({"SPY": -0.002, "QQQ": -0.001, "BTC": -0.001, "ETH": -0.001})
        prices.iloc[-1] = prices.iloc[-2] * 5

        result = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            fee_rate=0.001,
            start_date=prices.index[-1],
        )

        first_allocation = result.allocations.iloc[0]
        self.assertEqual(first_allocation["cash"], 1.0)
        self.assertEqual(sum(first_allocation[asset] for asset in RISK_ASSETS), 0.0)
        self.assertEqual(result.decisions[0].asof_date, prices.index[-2])
        self.assertAlmostEqual(result.final_equity, 10000.0)

    def test_allocations_always_include_all_assets(self):
        prices = trend_prices({"SPY": 0.001, "QQQ": 0.002, "BTC": 0.003, "ETH": 0.004})

        result = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            start_date=prices.index[-5],
        )

        self.assertEqual(list(result.allocations.columns), list(ALL_ASSETS))
        for _, row in result.allocations.iterrows():
            self.assertAlmostEqual(float(row.sum()), 1.0)

    def test_crypto_cap_limits_btc_eth_weight(self):
        prices = trend_prices({"SPY": 0.001, "QQQ": 0.002, "BTC": 0.006, "ETH": 0.005})

        allocation, _, risk_on = select_allocation(
            prices,
            len(prices) - 1,
            strategy_config=AllocatorStrategyConfig(max_crypto_weight=0.4),
        )

        self.assertTrue(risk_on)
        self.assertAlmostEqual(allocation["BTC"] + allocation["ETH"], 0.4)
        self.assertAlmostEqual(allocation["cash"], 0.6)

    def test_asset_trend_filter_blocks_asset_below_own_sma(self):
        index = pd.date_range("2024-01-01", periods=230, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": np.full(230, 100.0),
                "QQQ": np.full(230, 100.0),
                "BTC": np.full(230, 400.0),
                "ETH": np.full(230, 100.0),
            },
            index=index,
        )
        prices.loc[prices.index[-181], "BTC"] = 100.0
        prices.loc[prices.index[-91], "BTC"] = 100.0
        prices.loc[prices.index[-1], "BTC"] = 350.0
        prices.loc[prices.index[-1], "SPY"] = 130.0
        prices.loc[prices.index[-1], "QQQ"] = 120.0
        prices.loc[prices.index[-1], "ETH"] = 110.0

        unfiltered, _, _ = select_allocation(prices, len(prices) - 1)
        filtered, _, _ = select_allocation(
            prices,
            len(prices) - 1,
            strategy_config=AllocatorStrategyConfig(asset_trend_filter=True),
        )

        self.assertGreater(unfiltered["BTC"], 0.0)
        self.assertEqual(filtered["BTC"], 0.0)

    def test_monthly_rebalance_has_fewer_decisions_than_weekly(self):
        prices = trend_prices({"SPY": 0.001, "QQQ": 0.002, "BTC": 0.003, "ETH": 0.004})

        weekly = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            start_date=prices.index[-80],
        )
        monthly = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            start_date=prices.index[-80],
            strategy_config=AllocatorStrategyConfig(rebalance="monthly"),
        )

        self.assertLess(len(monthly.decisions), len(weekly.decisions))

    def test_experiment_rows_include_candidate_metrics(self):
        prices = trend_prices({"SPY": 0.001, "QQQ": 0.002, "BTC": 0.003, "ETH": 0.004})

        rows = build_experiment_rows(prices, initial_capital=10000.0, days=120)
        names = [row["name"] for row in rows]

        self.assertIn("v1 weekly top2", names)
        self.assertIn("combo risk sized", names)
        self.assertIn("combo defensive", names)
        self.assertTrue(all("total" in row and "vs_spy" in row for row in rows))

    def test_risk_off_defensive_mode_uses_positive_defensive_asset(self):
        prices = trend_prices_with_defensive(
            {
                "SPY": -0.002,
                "QQQ": -0.002,
                "BTC": -0.003,
                "ETH": -0.003,
                "GLD": 0.001,
                "TLT": -0.001,
                "SHY": 0.0002,
                "BIL": 0.0001,
            }
        )

        allocation, _, risk_on = select_allocation(
            prices,
            len(prices) - 1,
            strategy_config=AllocatorStrategyConfig(defensive_mode="top1"),
        )

        self.assertFalse(risk_on)
        self.assertEqual(allocation["GLD"], 1.0)
        self.assertEqual(allocation["cash"], 0.0)

    def test_defensive_returns_are_included_in_equity_curve(self):
        prices = trend_prices_with_defensive(
            {
                "SPY": -0.002,
                "QQQ": -0.002,
                "BTC": -0.003,
                "ETH": -0.003,
                "GLD": 0.001,
                "TLT": -0.001,
                "SHY": 0.0002,
                "BIL": 0.0001,
            }
        )
        prices.iloc[-1, prices.columns.get_loc("GLD")] = (
            prices.iloc[-2, prices.columns.get_loc("GLD")] * 1.01
        )

        result = run_allocator_backtest(
            prices,
            initial_capital=10000.0,
            fee_rate=0.0,
            start_date=prices.index[-1],
            strategy_config=AllocatorStrategyConfig(defensive_mode="top1"),
        )

        self.assertEqual(result.allocations.iloc[0]["GLD"], 1.0)
        self.assertAlmostEqual(result.final_equity, 10100.0)

    def test_rolling_window_rows_summarize_stability(self):
        prices = trend_prices_with_defensive(
            {
                "SPY": 0.001,
                "QQQ": 0.002,
                "BTC": 0.003,
                "ETH": 0.002,
                "GLD": 0.0005,
                "TLT": 0.0001,
                "SHY": 0.0001,
                "BIL": 0.0001,
            },
            rows=620,
        )

        rows = build_rolling_window_rows(
            prices,
            initial_capital=10000.0,
            window_days=120,
            step_days=60,
            configs=[
                AllocatorStrategyConfig(name="v1 weekly top2"),
                AllocatorStrategyConfig(name="defensive", defensive_mode="top1"),
            ],
        )

        self.assertEqual([row["name"] for row in rows], ["v1 weekly top2", "defensive"])
        self.assertTrue(all(row["windows"] > 0 for row in rows))
        self.assertTrue(all(0.0 <= row["win_spy_pct"] <= 100.0 for row in rows))
        self.assertTrue(all("worst_vs_spy" in row for row in rows))

    def test_validation_sections_return_rows(self):
        prices = trend_prices_with_defensive(
            {
                "SPY": 0.0005,
                "QQQ": 0.0007,
                "BTC": 0.001,
                "ETH": 0.0008,
                "GLD": 0.0003,
                "TLT": 0.0001,
                "SHY": 0.0001,
                "BIL": 0.0001,
            },
            rows=1800,
        )
        prices.index = pd.date_range("2017-01-02", periods=len(prices), freq="B")

        benchmark_rows = build_candidate_benchmark_rows(
            prices,
            initial_capital=10000.0,
            days=365,
        )
        stress_rows = build_stress_period_rows(prices, initial_capital=10000.0)
        walk_forward_rows = build_walk_forward_rows(
            prices,
            initial_capital=10000.0,
            start_year=2018,
        )

        self.assertIn("riskoff defensive top1", [row["name"] for row in benchmark_rows])
        self.assertIn("simple monthly momentum", [row["name"] for row in benchmark_rows])
        self.assertTrue(any(row["label"] == "2020 COVID crash" for row in stress_rows))
        self.assertTrue(any(row["year"] == 2022 for row in walk_forward_rows))

    def test_extended_etf_rows_do_not_require_crypto_prices(self):
        prices = trend_prices_with_defensive(
            {
                "SPY": 0.0005,
                "QQQ": 0.0007,
                "BTC": 0.001,
                "ETH": 0.0008,
                "GLD": 0.0003,
                "TLT": 0.0001,
                "SHY": 0.0001,
                "BIL": 0.0001,
            },
            rows=620,
        ).drop(columns=["BTC", "ETH"])

        rows = build_extended_etf_rows(prices, initial_capital=10000.0)

        self.assertIn("ETF riskoff defensive", [row["name"] for row in rows])
        self.assertIn("ETF monthly momentum", [row["name"] for row in rows])


if __name__ == "__main__":
    unittest.main()
