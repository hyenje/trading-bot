import unittest

import pandas as pd

from backtesting.engine import BacktestEngine
from config import BacktestConfig
from strategies.base import BaseStrategy, Signal, TradeSignal


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


if __name__ == "__main__":
    unittest.main()
