# Crypto Trading Bot

Binance spot/futures testnet trading bot experiment with a Flask dashboard, dry-run mode, long/short signal observation, and focused backtesting tools.

This project should be treated as a simulation, backtesting, and engineering-learning project. Do not use it as a real-money trading system without separate production-grade review, monitoring, and risk controls.

## Setup

```bash
cd /Users/sinhyeonjae/Documents/money/crypto-trading-bot
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with the matching Binance key type for the mode you are testing. Spot testnet, futures testnet, and mainnet keys are different.

## Safety Defaults

`.env.example` starts in safe mode:

```bash
USE_TESTNET=true
DRY_RUN=true
ALLOW_LIVE_TRADING=false
ENABLE_LONG_SHORT_EXECUTION=false
```

General spot live trading is blocked unless `USE_TESTNET=false`, `DRY_RUN=false`, and `ALLOW_LIVE_TRADING=true` are all set. BTC long/short futures execution is testnet-only and additionally requires `ENABLE_LONG_SHORT_EXECUTION=true` with `DRY_RUN=false`.

## Checks And Runs

```bash
# Read-only spot settings/API check
./.venv/bin/python main.py --check

# Read-only Binance futures testnet check
./.venv/bin/python main.py --check-futures

# BTC long/short backtest using the current 10m/4h MTF default
./.venv/bin/python main.py --backtest-long-short

# Market regime allocator backtest using public Yahoo/Binance data
./.venv/bin/python main.py --backtest-allocator

# Observe long/short signals without futures orders
./.venv/bin/python main.py --observe-long-short

# Run BTC long/short futures testnet execution after enabling it in .env
./.venv/bin/python main.py --trade-long-short
```

Dashboard:

```text
http://127.0.0.1:5001
http://127.0.0.1:5001/api/status
```

The dashboard distinguishes raw 10m signals, executable signals, 4h regime, and
actual positions. A strategy-side `HOLD` with `LONG_BIAS` or `SHORT_BIAS` is
displayed as `WAIT_LONG_BIAS` or `WAIT_SHORT_BIAS`; it is still not an open
position. If the raw 10m signal does not match the 4h regime, the executable
side remains `HOLD` and the block reason is shown in the signal card.

If `LONG_SHORT_ENABLE_SIGNAL_CATCHUP=true`, the futures testnet executor may
enter on the latest recent signal after a restart, but only when the account is
flat, the signal is within `LONG_SHORT_MAX_SIGNAL_AGE_MINUTES`, and the current
4h regime still points in the same direction. The default is `false`.

The default BTC long/short profile is `10m` entry timing with a closed `4h`
regime filter. Reverse-signal closes/flips are held until estimated net PnL is
at least `BTC_LS_MIN_REVERSE_NET_PNL_USDT`; stop loss, take profit, and daily
loss safety exits still override that rule.

## Running In A Screen Session

```bash
screen -dmS btc-testnet bash -lc 'cd /Users/sinhyeonjae/Documents/money/crypto-trading-bot && ./.venv/bin/python main.py --trade-long-short >> logs/long-short-testnet.log 2>&1'
screen -ls
screen -S btc-testnet -X quit
```

If a child process survives after quitting the screen session, inspect it before terminating:

```bash
ps -eo pid,etime,command | rg "(crypto-trading-bot|main\\.py|btc-testnet)"
```

## Tests

No extra test dependency is required.

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Current Backtest Limits

The long/short backtest uses `LONG_SHORT_ORDER_USDT` as fixed notional sizing
and prints the current `10m/4h fee-aware reverse` strategy first, with the old
raw executor-like result and direction-only rows kept as baselines. The report
separates gross PnL, fees, and net PnL, then summarizes exit reasons, holding
time, regime-blocked signals, and reverse-blocked signals. The same command
also prints execution-assumption checks, fee/slippage stress rows, equal-period
OOS windows, fixed-parameter walk-forward holdouts, parameter sensitivity,
timeframe-filter comparisons, monthly PnL, regime/volatility breakdowns, and a
trade-distribution summary.

The active baseline exit profile is `LONG_SHORT_STOP_LOSS_PCT=2.0`,
`LONG_SHORT_TAKE_PROFIT_PCT=4.0`, `LONG_SHORT_MAX_HOLD_BARS=0`, and
`LONG_SHORT_BREAK_EVEN_AFTER_PCT=0.0`; `0` disables the optional time-exit and
break-even exits.

It still does not model funding, dynamic order-book slippage, partial fills,
latency, liquidation mechanics, or websocket execution timing. Treat good
results as observation evidence, not production proof, especially while the
current MTF profile has a small trade sample.

Binance Futures testnet rejects new non-reduce-only orders below the minimum
notional. Keep `LONG_SHORT_ORDER_USDT` above that threshold; the default example
uses `$60` so BTCUSDT testnet entries clear the `$50` minimum.

Current MTF watch points:

- The current profile is the default because it cut overtrading sharply versus
  the raw 10m executor-like baseline in the latest local backtest.
- The main concern is sample size: the latest default result had fewer than 30
  trades, so the win rate and profit factor can still be luck-heavy.
- Funding is less urgent in the latest BTCUSDT window, but long multi-day holds
  mean it should be monitored before trusting longer testnet results.
- Constant slippage stress is modeled, but dynamic slippage and partial fills
  are still not, so live/testnet observation matters more than another round of
  parameter tuning.

## Log Hygiene

Logs are ignored by git. Runtime logging masks configured secrets and signed query fields such as signatures, timestamps, recv windows, API keys, secrets, and tokens. If you inspect old logs after credential work, verify that sensitive query fragments are gone:

```bash
rg "signature=|apiKey=|secret=" logs
```
