import unittest

import pandas as pd

from backtesting.engine import BacktestEngine
from config import BTCTrendLongShortConfig, BacktestConfig
from strategies.base import BaseStrategy, Signal, TradeSignal
from strategies.btc_mtf_regime import build_regime_series


class OneShotStrategy(BaseStrategy):
    def __init__(self, signal):
        super().__init__("OneShot")
        self.signal = signal

    def get_indicators(self, df):
        return df

    def analyze(self, df, symbol):
        signal = self.signal if len(df) == 51 else Signal.HOLD
        return TradeSignal(
            signal=signal,
            symbol=symbol,
            strategy_name=self.name,
            confidence=1.0,
            price=df["close"].iloc[-1],
            reason="test",
        )


class SequenceStrategy(BaseStrategy):
    def __init__(self, signals):
        super().__init__("Sequence")
        self.signals = signals

    def get_indicators(self, df):
        return df

    def analyze(self, df, symbol):
        signal = self.signals.get(len(df), Signal.HOLD)
        return TradeSignal(
            signal=signal,
            symbol=symbol,
            strategy_name=self.name,
            confidence=1.0,
            price=df["close"].iloc[-1],
            reason=f"signal {signal.value}",
        )


def make_price_data(start=100.0, step=-1.0, rows=80):
    index = pd.date_range("2026-01-01", periods=rows, freq="h")
    close = pd.Series([start + i * step for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1000,
        }
    )


class BacktestLongShortTest(unittest.TestCase):
    def test_sell_signal_can_open_profitable_short_when_allowed(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(),
            OneShotStrategy(Signal.SELL),
            allow_short=True,
            stop_loss_pct=99,
            take_profit_pct=99,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].side, "short")
        self.assertGreater(result.trades[0].pnl, 0)

    def test_sell_signal_does_not_open_short_by_default(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(),
            OneShotStrategy(Signal.SELL),
            allow_short=False,
            stop_loss_pct=99,
            take_profit_pct=99,
        )

        self.assertEqual(result.total_trades, 0)

    def test_buy_signal_opens_existing_long_behavior(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            stop_loss_pct=99,
            take_profit_pct=99,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].side, "long")
        self.assertGreater(result.trades[0].pnl, 0)

    def test_fixed_notional_uses_order_sized_position(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertAlmostEqual(
            result.trades[0].amount * result.trades[0].entry_price,
            25.0,
        )
        self.assertEqual(result.capital_deployed_per_trade, 25.0)

    def test_flip_on_reverse_opens_opposite_side_after_close(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=-1.0),
            SequenceStrategy({51: Signal.SELL, 56: Signal.BUY}),
            allow_short=True,
            flip_on_reverse=True,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 2)
        self.assertEqual([trade.side for trade in result.trades], ["short", "long"])
        self.assertTrue(result.flip_on_reverse)

    def test_default_reverse_signal_only_closes_existing_position(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=-1.0),
            SequenceStrategy({51: Signal.SELL, 56: Signal.BUY}),
            allow_short=True,
            flip_on_reverse=False,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].side, "short")

    def test_pnl_splits_gross_fees_and_net(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.001))
        result = engine.run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        trade = result.trades[0]
        self.assertGreater(trade.gross_pnl, trade.pnl)
        self.assertAlmostEqual(trade.gross_pnl - trade.fee_paid, trade.pnl)
        self.assertAlmostEqual(result.gross_pnl - result.total_fees, result.total_pnl)

    def test_slippage_reduces_net_pnl(self):
        no_slippage = BacktestEngine(
            BacktestConfig(commission_rate=0.0, slippage_rate=0.0)
        ).run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )
        with_slippage = BacktestEngine(
            BacktestConfig(commission_rate=0.0, slippage_rate=0.01)
        ).run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(with_slippage.slippage_rate, 0.01)
        self.assertLess(with_slippage.total_pnl, no_slippage.total_pnl)

    def test_allowed_sides_blocks_new_long_entries(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=1.0),
            OneShotStrategy(Signal.BUY),
            allowed_sides="short",
            allow_short=True,
            stop_loss_pct=99,
            take_profit_pct=99,
        )

        self.assertEqual(result.total_trades, 0)

    def test_max_hold_bars_closes_position_by_time(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=0.0),
            OneShotStrategy(Signal.BUY),
            max_hold_bars=3,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].reason_exit, "시간 청산")
        self.assertEqual(result.exit_reason_counts["time_exit"], 1)

    def test_max_hold_zero_disables_time_exit(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=0.0),
            OneShotStrategy(Signal.BUY),
            max_hold_bars=0,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].reason_exit, "백테스트 종료")
        self.assertNotIn("time_exit", result.exit_reason_counts)

    def test_break_even_closes_after_profit_retrace(self):
        df = make_price_data(start=100.0, step=0.0, rows=60)
        df.loc[df.index[51], ["open", "high", "low", "close"]] = [
            100.2,
            100.6,
            100.2,
            100.6,
        ]
        df.loc[df.index[52], ["open", "high", "low", "close"]] = [
            100.2,
            100.2,
            99.9,
            100.0,
        ]

        result = BacktestEngine(BacktestConfig(commission_rate=0.0)).run(
            df,
            OneShotStrategy(Signal.BUY),
            break_even_after_pct=0.5,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].reason_exit, "본전 청산")
        self.assertEqual(result.exit_reason_counts["break_even"], 1)
        self.assertAlmostEqual(result.trades[0].pnl, 0.0)

    def test_regime_series_uses_only_closed_higher_timeframe_candles(self):
        index = pd.date_range("2026-01-01", periods=50, freq="10min")
        close = pd.Series(100.0, index=index)
        close.loc["2026-01-01 08:00":] = 200.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
            }
        )

        regime = build_regime_series(
            df,
            "4h",
            BTCTrendLongShortConfig(fast_ema=2, slow_ema=3, slope_period=1, rsi_period=2),
        )

        self.assertEqual(
            regime.loc[pd.Timestamp("2026-01-01 08:10"), "regime_closed_at"],
            pd.Timestamp("2026-01-01 08:00"),
        )

    def test_regime_mismatch_blocks_raw_entry_signal(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.0))
        result = engine.run(
            make_price_data(start=100.0, step=-1.0, rows=80),
            OneShotStrategy(Signal.BUY),
            allow_short=True,
            stop_loss_pct=99,
            take_profit_pct=99,
            regime_timeframe="4h",
            require_regime_alignment=True,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.blocked_by_regime_count, 1)

    def test_fee_aware_reverse_blocks_losing_position(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.001))
        result = engine.run(
            make_price_data(start=100.0, step=1.0, rows=80),
            SequenceStrategy({51: Signal.SELL, 56: Signal.BUY}),
            allow_short=True,
            flip_on_reverse=True,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
            reverse_only_when_profitable=True,
            min_reverse_net_pnl_usdt=0.0,
        )

        self.assertEqual(result.reverse_block_count, 1)
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].side, "short")
        self.assertEqual(result.trades[0].reason_exit, "백테스트 종료")

    def test_fee_aware_reverse_allows_profitable_position(self):
        engine = BacktestEngine(BacktestConfig(commission_rate=0.001))
        result = engine.run(
            make_price_data(start=100.0, step=-1.0, rows=80),
            SequenceStrategy({51: Signal.SELL, 56: Signal.BUY}),
            allow_short=True,
            flip_on_reverse=True,
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
            reverse_only_when_profitable=True,
            min_reverse_net_pnl_usdt=0.0,
        )

        self.assertEqual(result.reverse_block_count, 0)
        self.assertEqual(result.total_trades, 2)
        self.assertEqual([trade.side for trade in result.trades], ["short", "long"])


if __name__ == "__main__":
    unittest.main()
