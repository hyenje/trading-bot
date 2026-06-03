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

# BTC long/short backtest using executor-like fixed notional sizing
./.venv/bin/python main.py --backtest-long-short

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

The dashboard distinguishes signal bias from actual positions. A strategy-side
`HOLD` with `LONG_BIAS` or `SHORT_BIAS` is displayed as `WAIT_LONG_BIAS` or
`WAIT_SHORT_BIAS`; it is still not an open position.

If `LONG_SHORT_ENABLE_SIGNAL_CATCHUP=true`, the futures testnet executor may
enter on the latest recent signal after a restart, but only when the account is
flat, the signal is within `LONG_SHORT_MAX_SIGNAL_AGE_MINUTES`, and the current
bias still points in the same direction. The default is `false`.

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

The long/short backtest now uses `LONG_SHORT_ORDER_USDT` as fixed notional sizing, closes and flips on reverse signals by default in `--backtest-long-short`, and prints long-only, short-only, and both-direction results together. The report separates gross PnL, fees, and net PnL, then summarizes exit reasons and holding time. The same command also runs a small fixed set of diagnostics for a stricter long filter, time exits, higher-timeframe trend filtering, EMA gap, and EMA slope.

It still does not model funding, slippage, partial fills, latency, liquidation mechanics, or websocket execution timing. Treat good results as observation evidence, not production proof.

## Log Hygiene

Logs are ignored by git. Runtime logging masks configured secrets and signed query fields such as signatures, timestamps, recv windows, API keys, secrets, and tokens. If you inspect old logs after credential work, verify that sensitive query fragments are gone:

```bash
rg "signature=|apiKey=|secret=" logs
```
