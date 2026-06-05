import unittest

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from config import BTCRegimePullbackConfig, BacktestConfig
from strategies.base import BaseStrategy, Signal, TradeSignal
from strategies.btc_regime_pullback import (
    BTCRegimePullbackStrategy,
    RANGE,
    TREND_DOWN,
    TREND_UP,
    UNKNOWN,
    build_regime_state_series,
    classify_regime,
)


def make_ohlcv(rows=80, start=100.0, step=0.0, freq="15min"):
    index = pd.date_range("2026-01-01", periods=rows, freq=freq)
    close = pd.Series([start + i * step for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000,
        }
    )


class InjectedRegimePullbackStrategy(BTCRegimePullbackStrategy):
    def __init__(self, config, regime_state, previous_row, current_row):
        super().__init__(config)
        self.regime_state = regime_state
        self.previous_row = previous_row
        self.current_row = current_row

    def get_indicators(self, df):
        data = make_ohlcv(rows=len(df), freq="15min")
        data.index = df.index
        data["rsi"] = 50.0
        data["bb_lower"] = 95.0
        data["bb_middle"] = 100.0
        data["bb_upper"] = 105.0
        data["regime_state"] = self.regime_state
        data["regime_trend_gap"] = 0.01
        data["regime_slope_norm"] = 0.001
        data["regime_closed_at"] = df.index
        for key, value in self.previous_row.items():
            data.iloc[-2, data.columns.get_loc(key)] = value
        for key, value in self.current_row.items():
            data.iloc[-1, data.columns.get_loc(key)] = value
        return data


class MetadataExitStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MetadataExit")

    def get_indicators(self, df):
        return df

    def analyze(self, df, symbol):
        signal = Signal.BUY if len(df) == 51 else Signal.HOLD
        return TradeSignal(
            signal=signal,
            symbol=symbol,
            strategy_name=self.name,
            confidence=1.0,
            price=df["close"].iloc[-1],
            reason="metadata exit",
            metadata={
                "exit_stop_loss_pct": 1.0,
                "exit_take_profit_pct": 99.0,
                "exit_max_hold_bars": 0,
            },
        )


class BTCRegimePullbackTest(unittest.TestCase):
    def test_regime_classifier_returns_all_states(self):
        config = BTCRegimePullbackConfig(regime_min_gap_pct=0.003)

        self.assertEqual(
            classify_regime(
                pd.Series(
                    {
                        "ema_fast": 102,
                        "ema_slow": 100,
                        "regime_trend_gap": 0.02,
                        "regime_slope_norm": 0.001,
                    }
                ),
                config,
            ),
            TREND_UP,
        )
        self.assertEqual(
            classify_regime(
                pd.Series(
                    {
                        "ema_fast": 98,
                        "ema_slow": 100,
                        "regime_trend_gap": -0.02,
                        "regime_slope_norm": -0.001,
                    }
                ),
                config,
            ),
            TREND_DOWN,
        )
        self.assertEqual(
            classify_regime(
                pd.Series(
                    {
                        "ema_fast": 100.1,
                        "ema_slow": 100,
                        "regime_trend_gap": 0.001,
                        "regime_slope_norm": 0.001,
                    }
                ),
                config,
            ),
            RANGE,
        )
        self.assertEqual(
            classify_regime(
                pd.Series(
                    {
                        "ema_fast": np.nan,
                        "ema_slow": 100,
                        "regime_trend_gap": np.nan,
                        "regime_slope_norm": np.nan,
                    }
                ),
                config,
            ),
            UNKNOWN,
        )

    def test_trend_up_allows_only_long_pullback(self):
        long_strategy = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            TREND_UP,
            {"close": 94.5, "rsi": 32.0},
            {"close": 96.0, "rsi": 34.0},
        )
        short_setup = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            TREND_UP,
            {"close": 106.0, "rsi": 75.0},
            {"close": 104.0, "rsi": 68.0},
        )

        self.assertEqual(long_strategy.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.BUY)
        self.assertEqual(short_setup.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.HOLD)

    def test_trend_down_allows_only_short_rally(self):
        short_strategy = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            TREND_DOWN,
            {"close": 106.0, "rsi": 75.0},
            {"close": 104.0, "rsi": 68.0},
        )
        long_setup = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            TREND_DOWN,
            {"close": 94.5, "rsi": 32.0},
            {"close": 96.0, "rsi": 34.0},
        )

        self.assertEqual(short_strategy.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.SELL)
        self.assertEqual(long_setup.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.HOLD)

    def test_range_allows_both_mean_reversion_directions(self):
        long_strategy = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            RANGE,
            {"close": 94.5, "rsi": 32.0},
            {"close": 96.0, "rsi": 34.0},
        )
        short_strategy = InjectedRegimePullbackStrategy(
            BTCRegimePullbackConfig(mode="combined"),
            RANGE,
            {"close": 106.0, "rsi": 75.0},
            {"close": 104.0, "rsi": 68.0},
        )

        self.assertEqual(long_strategy.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.BUY)
        self.assertEqual(short_strategy.analyze(make_ohlcv(), "BTC/USDT").signal, Signal.SELL)

    def test_regime_series_uses_only_closed_higher_timeframe_candles(self):
        df = make_ohlcv(rows=40, start=100.0, step=0.0, freq="15min")
        df.loc["2026-01-01 08:15":, "close"] = 140.0
        df["open"] = df["close"]
        df["high"] = df["close"] + 0.2
        df["low"] = df["close"] - 0.2

        regime = build_regime_state_series(
            df,
            BTCRegimePullbackConfig(
                fast_ema=2,
                slow_ema=3,
                slope_period=1,
                regime_min_gap_pct=0.0001,
            ),
        )

        latest = regime.loc[pd.Timestamp("2026-01-01 09:45")]
        self.assertLessEqual(latest["regime_closed_at"], pd.Timestamp("2026-01-01 09:45"))
        self.assertNotEqual(latest["regime_closed_at"], pd.Timestamp("2026-01-01 12:00"))

    def test_backtest_uses_signal_metadata_exit_profile(self):
        df = make_ohlcv(rows=60, start=100.0, step=0.0, freq="1h")
        df.loc[df.index[51], "low"] = 98.8

        result = BacktestEngine(BacktestConfig(commission_rate=0.0)).run(
            df,
            MetadataExitStrategy(),
            stop_loss_pct=99,
            take_profit_pct=99,
            position_size_usdt=25.0,
        )

        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.trades[0].reason_exit, "손절")


if __name__ == "__main__":
    unittest.main()
