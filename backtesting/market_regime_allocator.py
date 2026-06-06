"""
Market regime allocator backtest.

This module is intentionally separate from the live bot/executor path. It only
uses public market data and simulates weekly portfolio allocation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests


RISK_ASSETS = ("SPY", "QQQ", "BTC", "ETH")
DEFENSIVE_ASSETS = ("GLD", "TLT", "SHY", "BIL")
TRADABLE_ASSETS = RISK_ASSETS + DEFENSIVE_ASSETS
MACRO_ASSETS = ("VIX",)
PRICE_ASSETS = TRADABLE_ASSETS + MACRO_ASSETS
ALL_ASSETS = TRADABLE_ASSETS + ("cash",)
CRYPTO_ASSETS = ("BTC", "ETH")
DEFAULT_DAYS = 1095
EXTENDED_VALIDATION_DAYS = 3300
LOOKBACK_DAYS = 365
REBALANCE_FEE_RATE = 0.001
MOMENTUM_FAST_DAYS = 90
MOMENTUM_SLOW_DAYS = 180
SPY_SMA_DAYS = 200
VOL_LOOKBACK_DAYS = 60
VIX_STRESS_LEVEL = 25.0
VIX_STRESS_RATIO = 0.25
TLT_STRESS_LOOKBACK_DAYS = 90
TLT_STRESS_RETURN = -0.07
EXTENDED_ETF_START = "2010-01-01"
STRESS_PERIODS = (
    ("2018 Q4 risk-off", "2018-09-01", "2018-12-31"),
    ("2020 COVID crash", "2020-02-01", "2020-04-30"),
    ("2022 inflation bear", "2022-01-01", "2022-12-31"),
)


@dataclass
class AllocatorStrategyConfig:
    name: str = "v1 weekly top2"
    risk_assets: Tuple[str, ...] = RISK_ASSETS
    defensive_assets: Tuple[str, ...] = DEFENSIVE_ASSETS
    score_mode: str = "momentum"
    risk_mode: str = "spy_200"
    asset_trend_filter: bool = False
    weighting: str = "top2_equal"
    rebalance: str = "weekly"
    rebalance_threshold: float = 0.0
    max_crypto_weight: Optional[float] = None
    max_single_asset: Optional[float] = None
    defensive_mode: str = "cash"
    macro_stress_mode: str = "none"
    macro_assets: Tuple[str, ...] = ()


@dataclass
class AllocationDecision:
    date: pd.Timestamp
    asof_date: pd.Timestamp
    risk_on: bool
    allocation: Dict[str, float]
    scores: Dict[str, float]


@dataclass
class PortfolioResult:
    name: str
    initial_capital: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    equity_curve: pd.Series
    daily_returns: pd.Series
    turnover: float = 0.0
    total_cost: float = 0.0
    allocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: List[AllocationDecision] = field(default_factory=list)
    asset_week_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return self.initial_capital
        return float(self.equity_curve.iloc[-1])

    @property
    def final_allocation(self) -> Dict[str, float]:
        if self.allocations.empty:
            return _cash_allocation()
        return {
            asset: float(self.allocations.iloc[-1].get(asset, 0.0))
            for asset in ALL_ASSETS
        }


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    calmar: float
    final_equity: float
    turnover: float


@dataclass
class AllocatorReport:
    prices: pd.DataFrame
    allocator: PortfolioResult
    benchmarks: Dict[str, PortfolioResult]
    metrics: Dict[str, PerformanceMetrics]
    oos_rows: List[Dict[str, float]]
    days: int
    experiment_rows: List[Dict[str, float]] = field(default_factory=list)
    recent_experiment_rows: List[Dict[str, float]] = field(default_factory=list)
    rolling_window_rows: List[Dict[str, float]] = field(default_factory=list)
    candidate_benchmark_rows: List[Dict[str, float]] = field(default_factory=list)
    stress_period_rows: List[Dict[str, float]] = field(default_factory=list)
    walk_forward_rows: List[Dict[str, float]] = field(default_factory=list)
    extended_etf_rows: List[Dict[str, float]] = field(default_factory=list)


def fetch_market_regime_prices(days: int = DEFAULT_DAYS) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days + LOOKBACK_DAYS + 10)

    series = {
        "SPY": fetch_yahoo_adjusted_close("SPY", start, end),
        "QQQ": fetch_yahoo_adjusted_close("QQQ", start, end),
        "GLD": fetch_yahoo_adjusted_close("GLD", start, end),
        "TLT": fetch_yahoo_adjusted_close("TLT", start, end),
        "SHY": fetch_yahoo_adjusted_close("SHY", start, end),
        "BIL": fetch_yahoo_adjusted_close("BIL", start, end),
        "VIX": fetch_yahoo_adjusted_close("^VIX", start, end),
        "BTC": fetch_binance_daily_close("BTC/USDT", start, end),
        "ETH": fetch_binance_daily_close("ETH/USDT", start, end),
    }
    prices = pd.DataFrame(series).sort_index().dropna()
    if len(prices) <= SPY_SMA_DAYS:
        raise RuntimeError(
            f"Not enough aligned daily data: {len(prices)} rows after inner join"
        )
    return prices


def fetch_extended_etf_prices(start: str = EXTENDED_ETF_START) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(start)
    series = {
        "SPY": fetch_yahoo_adjusted_close("SPY", start_ts, end),
        "QQQ": fetch_yahoo_adjusted_close("QQQ", start_ts, end),
        "GLD": fetch_yahoo_adjusted_close("GLD", start_ts, end),
        "TLT": fetch_yahoo_adjusted_close("TLT", start_ts, end),
        "SHY": fetch_yahoo_adjusted_close("SHY", start_ts, end),
        "BIL": fetch_yahoo_adjusted_close("BIL", start_ts, end),
    }
    prices = pd.DataFrame(series).sort_index().dropna()
    if len(prices) <= SPY_SMA_DAYS:
        raise RuntimeError(
            f"Not enough aligned ETF data: {len(prices)} rows after inner join"
        )
    return prices


def fetch_yahoo_adjusted_close(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": _unix_seconds(start),
        "period2": _unix_seconds(end),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "crypto-trading-bot/allocator-backtest"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")

    data = result[0]
    timestamps = data.get("timestamp") or []
    indicators = data.get("indicators", {})
    adjclose = indicators.get("adjclose") or []
    quote = indicators.get("quote") or []
    values = []
    if adjclose:
        values = adjclose[0].get("adjclose") or []
    if not values and quote:
        values = quote[0].get("close") or []
    if not timestamps or not values:
        raise RuntimeError(f"Yahoo close data missing for {ticker}")

    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
    series = pd.Series(values, index=index, name=ticker, dtype="float64").dropna()
    return series.groupby(series.index).last()


def fetch_binance_daily_close(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {
                "fetchCurrencies": False,
            },
        }
    )
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows = []

    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + 24 * 60 * 60 * 1000
        if next_since <= since:
            break
        since = next_since
        if len(batch) < 1000:
            break

    if not rows:
        raise RuntimeError(f"Binance returned no data for {symbol}")

    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    index = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    index = index.dt.normalize()
    label = symbol.split("/")[0]
    series = pd.Series(df["close"].to_numpy(dtype=float), index=index, name=label)
    return series.groupby(series.index).last()


def run_allocator_backtest(
    prices: pd.DataFrame,
    initial_capital: float,
    fee_rate: float = REBALANCE_FEE_RATE,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    strategy_config: Optional[AllocatorStrategyConfig] = None,
) -> PortfolioResult:
    strategy_config = strategy_config or AllocatorStrategyConfig()
    prices = _clean_prices(
        prices,
        required_assets=strategy_config.risk_assets,
        asset_scope=_strategy_price_assets(strategy_config),
    )
    start_i, end_i = _date_window(prices, start_date, end_date)
    allocation = _cash_allocation()
    equity = float(initial_capital)
    equity_values = []
    daily_returns = []
    dates = []
    allocation_rows = []
    decisions = []
    total_cost = 0.0
    total_turnover = 0.0

    for i in range(start_i, end_i + 1):
        date = prices.index[i]
        equity_before_day = equity
        if i == start_i or _should_rebalance(
            prices.index[i - 1],
            date,
            strategy_config,
        ):
            target, scores, risk_on = select_allocation(
                prices,
                i - 1,
                strategy_config=strategy_config,
            )
            if (
                strategy_config.rebalance_threshold > 0
                and _risky_turnover(allocation, target)
                < strategy_config.rebalance_threshold
            ):
                target = allocation.copy()
            traded_ratio = _risky_turnover(allocation, target)
            equity_before_cost = equity
            cost = equity_before_cost * traded_ratio * fee_rate
            equity -= cost
            total_cost += cost
            total_turnover += traded_ratio
            allocation = target
            decisions.append(
                AllocationDecision(
                    date=date,
                    asof_date=prices.index[i - 1],
                    risk_on=risk_on,
                    allocation=target.copy(),
                    scores=scores,
                )
            )

        asset_returns = prices.iloc[i] / prices.iloc[i - 1] - 1
        portfolio_return = sum(
            allocation[asset] * float(asset_returns[asset])
            for asset in _priced_assets(prices)
        )
        equity *= 1 + portfolio_return

        dates.append(date)
        equity_values.append(equity)
        daily_returns.append(equity / equity_before_day - 1)
        allocation_rows.append(allocation.copy())

    equity_curve = pd.Series(equity_values, index=pd.Index(dates), name="allocator")
    daily_return_series = pd.Series(
        daily_returns, index=pd.Index(dates), name="allocator_returns"
    )
    allocations = pd.DataFrame(allocation_rows, index=pd.Index(dates), columns=ALL_ASSETS)
    return PortfolioResult(
        name=strategy_config.name,
        initial_capital=float(initial_capital),
        start_date=prices.index[start_i],
        end_date=prices.index[end_i],
        equity_curve=equity_curve,
        daily_returns=daily_return_series,
        turnover=total_turnover,
        total_cost=total_cost,
        allocations=allocations,
        decisions=decisions,
        asset_week_counts=_asset_week_counts(allocations),
    )


def select_allocation(
    prices: pd.DataFrame,
    asof_index: int,
    strategy_config: Optional[AllocatorStrategyConfig] = None,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    strategy_config = strategy_config or AllocatorStrategyConfig()
    prices = _clean_prices(
        prices,
        required_assets=strategy_config.risk_assets,
        asset_scope=_strategy_price_assets(strategy_config),
    )
    if asof_index < max(MOMENTUM_SLOW_DAYS, SPY_SMA_DAYS):
        return _cash_allocation(), {}, False

    history = prices.iloc[: asof_index + 1]
    scores = _momentum_scores(history, strategy_config)
    if _is_macro_stressed(history, strategy_config):
        allocation = _defensive_allocation(history, strategy_config)
        return allocation, scores, False

    risk_on = _is_risk_on(history, strategy_config)
    if not risk_on:
        allocation = _defensive_allocation(history, strategy_config)
        return allocation, scores, False

    positive = [
        (asset, score)
        for asset, score in scores.items()
        if score == score
        and score > 0
        and _is_asset_eligible(history, asset, strategy_config)
    ]
    positive.sort(key=lambda item: item[1], reverse=True)
    allocation = _build_target_allocation(positive, strategy_config)
    return allocation, scores, True


def run_buy_hold(
    prices: pd.DataFrame,
    asset: str,
    initial_capital: float,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> PortfolioResult:
    prices = _clean_prices(
        prices,
        required_assets=(asset,),
        asset_scope=TRADABLE_ASSETS,
    )
    start_i, end_i = _date_window(prices, start_date, end_date)
    returns = prices[asset].pct_change().iloc[start_i : end_i + 1]
    equity_curve = initial_capital * (1 + returns).cumprod()
    allocation = _cash_allocation()
    allocation[asset] = 1.0
    allocation["cash"] = 0.0
    allocations = pd.DataFrame(
        [allocation.copy() for _ in range(len(equity_curve))],
        index=equity_curve.index,
        columns=ALL_ASSETS,
    )
    return PortfolioResult(
        name=f"{asset} B&H",
        initial_capital=float(initial_capital),
        start_date=prices.index[start_i],
        end_date=prices.index[end_i],
        equity_curve=equity_curve.rename(f"{asset}_buy_hold"),
        daily_returns=returns.rename(f"{asset}_returns"),
        allocations=allocations,
        asset_week_counts=_asset_week_counts(allocations),
    )


def run_static_allocation(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    name: str,
    initial_capital: float,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> PortfolioResult:
    prices = _clean_prices(
        prices,
        required_assets=tuple(weights.keys()),
        asset_scope=TRADABLE_ASSETS,
    )
    start_i, end_i = _date_window(prices, start_date, end_date)
    returns = prices[list(weights.keys())].pct_change().iloc[start_i : end_i + 1]
    portfolio_returns = returns.mul(pd.Series(weights)).sum(axis=1)
    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()
    allocation = _cash_allocation()
    for asset, weight in weights.items():
        allocation[asset] = float(weight)
    allocation["cash"] = max(0.0, 1.0 - sum(weights.values()))
    allocations = pd.DataFrame(
        [allocation.copy() for _ in range(len(equity_curve))],
        index=equity_curve.index,
        columns=ALL_ASSETS,
    )
    return PortfolioResult(
        name=name,
        initial_capital=float(initial_capital),
        start_date=prices.index[start_i],
        end_date=prices.index[end_i],
        equity_curve=equity_curve.rename(name),
        daily_returns=portfolio_returns.rename(f"{name}_returns"),
        allocations=allocations,
        asset_week_counts=_asset_week_counts(allocations),
    )


def build_allocator_report(
    prices: pd.DataFrame,
    initial_capital: float,
    days: int = DEFAULT_DAYS,
    extended_etf_prices: Optional[pd.DataFrame] = None,
) -> AllocatorReport:
    prices = _clean_prices(prices)
    start_date = prices.index[-1] - pd.Timedelta(days=days)
    allocator = run_allocator_backtest(
        prices,
        initial_capital=initial_capital,
        start_date=start_date,
        strategy_config=tlt_stress_candidate_config(),
    )
    benchmarks = {
        asset: run_buy_hold(
            prices,
            asset,
            initial_capital=initial_capital,
            start_date=start_date,
        )
        for asset in ("SPY", "QQQ", "BTC")
    }
    results = {"Allocator": allocator}
    results.update({f"{asset} B&H": result for asset, result in benchmarks.items()})
    metrics = {
        name: calculate_metrics(result)
        for name, result in results.items()
    }
    return AllocatorReport(
        prices=prices.loc[allocator.start_date : allocator.end_date],
        allocator=allocator,
        benchmarks=benchmarks,
        metrics=metrics,
        oos_rows=_oos_rows(results),
        days=days,
        experiment_rows=build_experiment_rows(prices, initial_capital, days),
        recent_experiment_rows=build_experiment_rows(prices, initial_capital, 365),
        rolling_window_rows=build_rolling_window_rows(
            prices,
            initial_capital,
            start_date=start_date,
        ),
        candidate_benchmark_rows=build_candidate_benchmark_rows(
            prices,
            initial_capital,
            days,
        ),
        stress_period_rows=build_stress_period_rows(prices, initial_capital),
        walk_forward_rows=build_walk_forward_rows(prices, initial_capital),
        extended_etf_rows=build_extended_etf_rows(
            extended_etf_prices,
            initial_capital,
        )
        if extended_etf_prices is not None
        else [],
    )


def calculate_metrics(result: PortfolioResult) -> PerformanceMetrics:
    final_equity = result.final_equity
    total_return = final_equity / result.initial_capital - 1
    days = max((result.end_date - result.start_date).days, 1)
    cagr = (final_equity / result.initial_capital) ** (365.25 / days) - 1

    curve_values = np.array(
        [result.initial_capital] + result.equity_curve.astype(float).tolist()
    )
    peaks = np.maximum.accumulate(curve_values)
    drawdowns = curve_values / peaks - 1
    max_drawdown = float(drawdowns.min())

    returns = result.daily_returns.astype(float).dropna()
    if len(returns) > 1 and float(returns.std(ddof=0)) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return PerformanceMetrics(
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        max_drawdown_pct=max_drawdown * 100,
        sharpe=sharpe,
        calmar=calmar,
        final_equity=final_equity,
        turnover=result.turnover,
    )


def _metric_row(result: PortfolioResult) -> Dict[str, float]:
    metric = calculate_metrics(result)
    return {
        "name": result.name,
        "start": result.start_date.date().isoformat(),
        "end": result.end_date.date().isoformat(),
        "total": metric.total_return_pct,
        "cagr": metric.cagr_pct,
        "maxdd": metric.max_drawdown_pct,
        "sharpe": metric.sharpe,
        "final_eq": metric.final_equity,
    }


def format_allocator_report(report: AllocatorReport) -> str:
    lines = [
        "",
        f"=== Market Regime Allocator: {report.allocator.name} ===",
        (
            f"Data: {report.prices.index[0].date()} -> "
            f"{report.prices.index[-1].date()} ({len(report.prices)} trading days)"
        ),
        (
            "Sources: Yahoo Finance adjusted close "
            "(SPY/QQQ/GLD/TLT/SHY/BIL/VIX), Binance public daily OHLCV (BTC/ETH)"
        ),
        (
            "Rules: weekly rebalance, SPY > 200d SMA risk-on, "
            "score=0.6*90d+0.4*180d, force defensive when TLT 90d <= -7% "
            "and TLT < 200d SMA"
        ),
        f"Cost: {REBALANCE_FEE_RATE * 100:.2f}% on changed risky notional",
        "",
        "--- Performance ---",
        (
            "case             total%   CAGR%   maxDD%  Sharpe  Calmar "
            "final_eq turnover"
        ),
    ]
    for name in ("Allocator", "SPY B&H", "QQQ B&H", "BTC B&H"):
        metric = report.metrics[name]
        lines.append(
            f"{name[:15]:<15} "
            f"{metric.total_return_pct:>7.2f} "
            f"{metric.cagr_pct:>7.2f} "
            f"{metric.max_drawdown_pct:>7.2f} "
            f"{metric.sharpe:>7.2f} "
            f"{metric.calmar:>7.2f} "
            f"{metric.final_equity:>8.2f} "
            f"{metric.turnover:>7.2f}x"
        )

    final_allocation = report.allocator.final_allocation
    allocation_text = ", ".join(
        f"{asset}={final_allocation.get(asset, 0.0) * 100:.0f}%"
        for asset in ALL_ASSETS
        if final_allocation.get(asset, 0.0) > 0
    )
    week_counts = ", ".join(
        f"{asset}={report.allocator.asset_week_counts.get(asset, 0)}"
        for asset in ALL_ASSETS
    )
    lines.extend(
        [
            "",
            "--- Allocation ---",
            f"Final allocation: {allocation_text or 'cash=100%'}",
            f"Asset weeks: {week_counts}",
            f"Rebalance decisions: {len(report.allocator.decisions)}",
            f"Total rebalance cost: ${report.allocator.total_cost:.2f}",
            "",
            "--- 4-way OOS Split ---",
            "window                     days  Alloc%    SPY%    QQQ%    BTC% AllocDD%",
        ]
    )
    for row in report.oos_rows:
        lines.append(
            f"{row['label']:<26} "
            f"{int(row['days']):>4d} "
            f"{row['Allocator']:>7.2f} "
            f"{row['SPY B&H']:>7.2f} "
            f"{row['QQQ B&H']:>7.2f} "
            f"{row['BTC B&H']:>7.2f} "
            f"{row['Allocator_maxdd']:>8.2f}"
        )
    lines.extend(_format_experiment_rows("Experimental Variants", report.experiment_rows))
    lines.extend(
        _format_experiment_rows("Recent 1Y Variants", report.recent_experiment_rows)
    )
    lines.extend(_format_rolling_window_rows(report.rolling_window_rows))
    lines.extend(
        _format_candidate_benchmark_rows(report.candidate_benchmark_rows)
    )
    lines.extend(_format_stress_period_rows(report.stress_period_rows))
    lines.extend(_format_walk_forward_rows(report.walk_forward_rows))
    lines.extend(_format_extended_etf_rows(report.extended_etf_rows))
    return "\n".join(lines)


def build_rolling_window_rows(
    prices: pd.DataFrame,
    initial_capital: float,
    window_days: int = 365,
    step_days: int = 180,
    start_date: Optional[pd.Timestamp] = None,
    configs: Optional[List[AllocatorStrategyConfig]] = None,
) -> List[Dict[str, float]]:
    prices = _clean_prices(prices)
    configs = configs or allocator_experiment_configs()
    earliest_i = max(MOMENTUM_SLOW_DAYS, SPY_SMA_DAYS) + 1
    if len(prices) <= earliest_i + 2:
        return []

    start = prices.index[earliest_i]
    if start_date is not None:
        start = max(start, pd.Timestamp(start_date).normalize())
    latest_start = prices.index[-1] - pd.Timedelta(days=window_days)
    windows = []
    while start <= latest_start:
        end = start + pd.Timedelta(days=window_days)
        spy = run_buy_hold(prices, "SPY", initial_capital, start_date=start, end_date=end)
        if len(spy.equity_curve) > 20:
            windows.append((start, end, calculate_metrics(spy).total_return_pct))
        start += pd.Timedelta(days=step_days)
    if not windows:
        return []

    rows = []
    for config in configs:
        totals = []
        vs_spy = []
        max_drawdowns = []
        sharpes = []
        for start, end, spy_total in windows:
            result = run_allocator_backtest(
                prices,
                initial_capital=initial_capital,
                start_date=start,
                end_date=end,
                strategy_config=config,
            )
            metric = calculate_metrics(result)
            totals.append(metric.total_return_pct)
            vs_spy.append(metric.total_return_pct - spy_total)
            max_drawdowns.append(metric.max_drawdown_pct)
            sharpes.append(metric.sharpe)
        rows.append(
            {
                "name": config.name,
                "windows": len(totals),
                "avg_total": float(np.mean(totals)),
                "avg_vs_spy": float(np.mean(vs_spy)),
                "win_spy_pct": sum(1 for value in vs_spy if value > 0)
                / len(vs_spy)
                * 100,
                "positive_pct": sum(1 for value in totals if value > 0)
                / len(totals)
                * 100,
                "worst_total": float(min(totals)),
                "worst_vs_spy": float(min(vs_spy)),
                "worst_maxdd": float(min(max_drawdowns)),
                "avg_sharpe": float(np.mean(sharpes)),
            }
        )
    return rows


def build_candidate_benchmark_rows(
    prices: pd.DataFrame,
    initial_capital: float,
    days: int,
) -> List[Dict[str, float]]:
    prices = _clean_prices(prices)
    start_date = prices.index[-1] - pd.Timedelta(days=days)
    results = [
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=riskoff_defensive_candidate_config(),
        ),
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=tlt_stress_candidate_config(),
        ),
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=vix_tlt_stress_candidate_config(),
        ),
        run_buy_hold(prices, "SPY", initial_capital, start_date=start_date),
        run_buy_hold(prices, "QQQ", initial_capital, start_date=start_date),
        run_buy_hold(prices, "BTC", initial_capital, start_date=start_date),
        run_static_allocation(
            prices,
            {"SPY": 0.6, "TLT": 0.4},
            "60/40 SPY/TLT",
            initial_capital,
            start_date=start_date,
        ),
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=monthly_momentum_config(),
        ),
    ]
    return [_metric_row(result) for result in results]


def build_stress_period_rows(
    prices: pd.DataFrame,
    initial_capital: float,
) -> List[Dict[str, float]]:
    prices = _clean_prices(prices)
    rows = []
    for label, start, end in STRESS_PERIODS:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        if end_date < prices.index[0] or start_date > prices.index[-1]:
            continue
        candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=riskoff_defensive_candidate_config(),
        )
        macro_candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=tlt_stress_candidate_config(),
        )
        vix_tlt_candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=macro_stress_candidate_config(),
        )
        spy = run_buy_hold(
            prices,
            "SPY",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        btc = run_buy_hold(
            prices,
            "BTC",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        sixty_forty = run_static_allocation(
            prices,
            {"SPY": 0.6, "TLT": 0.4},
            "60/40 SPY/TLT",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        candidate_metric = calculate_metrics(candidate)
        macro_metric = calculate_metrics(macro_candidate)
        vix_tlt_metric = calculate_metrics(vix_tlt_candidate)
        rows.append(
            {
                "label": label,
                "start": candidate.start_date.date().isoformat(),
                "end": candidate.end_date.date().isoformat(),
                "candidate": candidate_metric.total_return_pct,
                "macro_candidate": macro_metric.total_return_pct,
                "vix_tlt_candidate": vix_tlt_metric.total_return_pct,
                "spy": calculate_metrics(spy).total_return_pct,
                "btc": calculate_metrics(btc).total_return_pct,
                "sixty_forty": calculate_metrics(sixty_forty).total_return_pct,
                "candidate_maxdd": candidate_metric.max_drawdown_pct,
                "macro_maxdd": macro_metric.max_drawdown_pct,
                "vix_tlt_maxdd": vix_tlt_metric.max_drawdown_pct,
                "candidate_sharpe": candidate_metric.sharpe,
                "macro_sharpe": macro_metric.sharpe,
            }
        )
    return rows


def build_walk_forward_rows(
    prices: pd.DataFrame,
    initial_capital: float,
    start_year: int = 2018,
) -> List[Dict[str, float]]:
    prices = _clean_prices(prices)
    rows = []
    final_year = prices.index[-1].year
    for year in range(start_year, final_year + 1):
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=12, day=31)
        if end_date < prices.index[0] or start_date > prices.index[-1]:
            continue
        candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=riskoff_defensive_candidate_config(),
        )
        macro_candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=tlt_stress_candidate_config(),
        )
        vix_tlt_candidate = run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
            strategy_config=macro_stress_candidate_config(),
        )
        if len(candidate.equity_curve) < 20:
            continue
        spy = run_buy_hold(
            prices,
            "SPY",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        btc = run_buy_hold(
            prices,
            "BTC",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        sixty_forty = run_static_allocation(
            prices,
            {"SPY": 0.6, "TLT": 0.4},
            "60/40 SPY/TLT",
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        candidate_metric = calculate_metrics(candidate)
        macro_metric = calculate_metrics(macro_candidate)
        vix_tlt_metric = calculate_metrics(vix_tlt_candidate)
        spy_total = calculate_metrics(spy).total_return_pct
        rows.append(
            {
                "year": year,
                "start": candidate.start_date.date().isoformat(),
                "end": candidate.end_date.date().isoformat(),
                "candidate": candidate_metric.total_return_pct,
                "macro_candidate": macro_metric.total_return_pct,
                "vix_tlt_candidate": vix_tlt_metric.total_return_pct,
                "spy": spy_total,
                "btc": calculate_metrics(btc).total_return_pct,
                "sixty_forty": calculate_metrics(sixty_forty).total_return_pct,
                "vs_spy": candidate_metric.total_return_pct - spy_total,
                "macro_vs_spy": macro_metric.total_return_pct - spy_total,
                "vix_tlt_vs_spy": vix_tlt_metric.total_return_pct - spy_total,
                "candidate_maxdd": candidate_metric.max_drawdown_pct,
                "macro_maxdd": macro_metric.max_drawdown_pct,
                "vix_tlt_maxdd": vix_tlt_metric.max_drawdown_pct,
            }
        )
    return rows


def build_extended_etf_rows(
    prices: pd.DataFrame,
    initial_capital: float,
) -> List[Dict[str, float]]:
    prices = _clean_prices(
        prices,
        required_assets=("SPY", "QQQ"),
        asset_scope=("SPY", "QQQ", "GLD", "TLT", "SHY", "BIL"),
    )
    start_date = prices.index[max(MOMENTUM_SLOW_DAYS, SPY_SMA_DAYS) + 1]
    results = [
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=etf_riskoff_defensive_config(),
        ),
        run_buy_hold(prices, "SPY", initial_capital, start_date=start_date),
        run_buy_hold(prices, "QQQ", initial_capital, start_date=start_date),
        run_static_allocation(
            prices,
            {"SPY": 0.6, "TLT": 0.4},
            "60/40 SPY/TLT",
            initial_capital,
            start_date=start_date,
        ),
        run_allocator_backtest(
            prices,
            initial_capital,
            start_date=start_date,
            strategy_config=etf_monthly_momentum_config(),
        ),
    ]
    return [_metric_row(result) for result in results]


def build_experiment_rows(
    prices: pd.DataFrame,
    initial_capital: float,
    days: int,
    configs: Optional[List[AllocatorStrategyConfig]] = None,
) -> List[Dict[str, float]]:
    prices = _clean_prices(prices)
    configs = configs or allocator_experiment_configs()
    start_date = prices.index[-1] - pd.Timedelta(days=days)
    spy = run_buy_hold(prices, "SPY", initial_capital, start_date=start_date)
    spy_metric = calculate_metrics(spy)
    rows = []
    for config in configs:
        result = run_allocator_backtest(
            prices,
            initial_capital=initial_capital,
            start_date=start_date,
            strategy_config=config,
        )
        metric = calculate_metrics(result)
        oos_positive, worst_oos = _oos_summary(result)
        rows.append(
            {
                "name": config.name,
                "total": metric.total_return_pct,
                "vs_spy": metric.total_return_pct - spy_metric.total_return_pct,
                "maxdd": metric.max_drawdown_pct,
                "sharpe": metric.sharpe,
                "turnover": metric.turnover,
                "cost": result.total_cost,
                "oos_positive": oos_positive,
                "worst_oos": worst_oos,
            }
        )
    return rows


def allocator_experiment_configs() -> List[AllocatorStrategyConfig]:
    return [
        AllocatorStrategyConfig(name="v1 weekly top2"),
        AllocatorStrategyConfig(
            name="asset trend filter",
            asset_trend_filter=True,
        ),
        AllocatorStrategyConfig(
            name="crypto cap 40",
            max_crypto_weight=0.4,
        ),
        AllocatorStrategyConfig(
            name="vol adjusted score",
            score_mode="vol_adjusted",
        ),
        AllocatorStrategyConfig(
            name="monthly rebalance",
            rebalance="monthly",
        ),
        AllocatorStrategyConfig(
            name="combo risk sized",
            score_mode="vol_adjusted",
            risk_mode="spy_qqq_200",
            asset_trend_filter=True,
            weighting="score_weighted",
            rebalance="weekly_threshold",
            rebalance_threshold=0.2,
            max_crypto_weight=0.4,
            max_single_asset=0.6,
        ),
        riskoff_defensive_candidate_config(),
        tlt_stress_candidate_config(),
        AllocatorStrategyConfig(
            name="combo defensive",
            score_mode="vol_adjusted",
            risk_mode="spy_qqq_200",
            asset_trend_filter=True,
            weighting="score_weighted",
            rebalance="weekly_threshold",
            rebalance_threshold=0.2,
            max_crypto_weight=0.4,
            max_single_asset=0.6,
            defensive_mode="top2_equal",
        ),
    ]


def riskoff_defensive_candidate_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="riskoff defensive top1",
        score_mode="vol_adjusted",
        defensive_mode="top1",
    )


def macro_stress_candidate_config() -> AllocatorStrategyConfig:
    return vix_tlt_stress_candidate_config()


def tlt_stress_candidate_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="tlt stress riskoff",
        score_mode="vol_adjusted",
        defensive_mode="top1",
        macro_stress_mode="tlt",
    )


def vix_tlt_stress_candidate_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="vix+tlt stress riskoff",
        score_mode="vol_adjusted",
        defensive_mode="top1",
        macro_stress_mode="vix_tlt",
        macro_assets=MACRO_ASSETS,
    )


def monthly_momentum_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="simple monthly momentum",
        risk_assets=TRADABLE_ASSETS,
        defensive_assets=(),
        risk_mode="always_on",
        weighting="top1",
        rebalance="monthly",
    )


def etf_riskoff_defensive_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="ETF riskoff defensive",
        risk_assets=("SPY", "QQQ"),
        defensive_assets=("GLD", "TLT", "SHY", "BIL"),
        score_mode="vol_adjusted",
        defensive_mode="top1",
    )


def etf_monthly_momentum_config() -> AllocatorStrategyConfig:
    return AllocatorStrategyConfig(
        name="ETF monthly momentum",
        risk_assets=("SPY", "QQQ", "GLD", "TLT", "SHY", "BIL"),
        defensive_assets=(),
        risk_mode="always_on",
        weighting="top1",
        rebalance="monthly",
    )


def _format_experiment_rows(title: str, rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        f"--- {title} ---",
        "case                    total%   vsSPY%   maxDD%  Sharpe turnover OOS+ worstOOS cost",
    ]
    for row in rows:
        lines.append(
            f"{row['name'][:22]:<22} "
            f"{row['total']:>7.2f} "
            f"{row['vs_spy']:>8.2f} "
            f"{row['maxdd']:>7.2f} "
            f"{row['sharpe']:>7.2f} "
            f"{row['turnover']:>7.2f}x "
            f"{int(row['oos_positive']):>4d}/4 "
            f"{row['worst_oos']:>8.2f} "
            f"{row['cost']:>5.0f}"
        )
    return lines


def _format_rolling_window_rows(rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        "--- Rolling 1Y Stability ---",
        "case                   windows   avg% avgVsSPY winSPY%   pos%  worst% worstVsSPY maxDD% Sharpe",
    ]
    if not rows:
        lines.append("not enough data")
        return lines
    for row in rows:
        lines.append(
            f"{row['name'][:22]:<22} "
            f"{int(row['windows']):>7d} "
            f"{row['avg_total']:>6.2f} "
            f"{row['avg_vs_spy']:>8.2f} "
            f"{row['win_spy_pct']:>7.0f} "
            f"{row['positive_pct']:>6.0f} "
            f"{row['worst_total']:>7.2f} "
            f"{row['worst_vs_spy']:>10.2f} "
            f"{row['worst_maxdd']:>6.2f} "
            f"{row['avg_sharpe']:>6.2f}"
        )
    return lines


def _format_candidate_benchmark_rows(rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        "--- Candidate Benchmarks ---",
        "case                    total%   CAGR%   maxDD%  Sharpe final_eq",
    ]
    for row in rows:
        lines.append(
            f"{row['name'][:22]:<22} "
            f"{row['total']:>7.2f} "
            f"{row['cagr']:>7.2f} "
            f"{row['maxdd']:>7.2f} "
            f"{row['sharpe']:>7.2f} "
            f"{row['final_eq']:>8.2f}"
        )
    return lines


def _format_stress_period_rows(rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        "--- Stress Regimes ---",
        "period                  window                    base     tlt vix+tlt    SPY    BTC  60/40  baseDD  tltDD vixDD",
    ]
    if not rows:
        lines.append("not enough data")
        return lines
    for row in rows:
        window = f"{row['start']}->{row['end']}"
        lines.append(
            f"{row['label'][:22]:<22} "
            f"{window:<22} "
            f"{row['candidate']:>7.2f} "
            f"{row['macro_candidate']:>7.2f} "
            f"{row['vix_tlt_candidate']:>7.2f} "
            f"{row['spy']:>6.2f} "
            f"{row['btc']:>6.2f} "
            f"{row['sixty_forty']:>6.2f} "
            f"{row['candidate_maxdd']:>7.2f} "
            f"{row['macro_maxdd']:>6.2f} "
            f"{row['vix_tlt_maxdd']:>5.2f}"
        )
    return lines


def _format_walk_forward_rows(rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        "--- Fixed-Parameter Walk-Forward ---",
        "year window                    base     tlt vix+tlt    SPY    BTC  60/40 baseVs tltVs vixVs baseDD tltDD vixDD",
    ]
    if not rows:
        lines.append("not enough data")
        return lines
    for row in rows:
        window = f"{row['start']}->{row['end']}"
        lines.append(
            f"{int(row['year']):>4d} "
            f"{window:<22} "
            f"{row['candidate']:>7.2f} "
            f"{row['macro_candidate']:>7.2f} "
            f"{row['vix_tlt_candidate']:>7.2f} "
            f"{row['spy']:>6.2f} "
            f"{row['btc']:>6.2f} "
            f"{row['sixty_forty']:>6.2f} "
            f"{row['vs_spy']:>7.2f} "
            f"{row['macro_vs_spy']:>5.2f} "
            f"{row['vix_tlt_vs_spy']:>5.2f} "
            f"{row['candidate_maxdd']:>7.2f} "
            f"{row['macro_maxdd']:>5.2f} "
            f"{row['vix_tlt_maxdd']:>5.2f}"
        )
    return lines


def _format_extended_etf_rows(rows: List[Dict[str, float]]) -> List[str]:
    lines = [
        "",
        "--- Extended ETF History ---",
        "case                    start       end         total%   CAGR%   maxDD%  Sharpe final_eq",
    ]
    if not rows:
        lines.append("not enough data")
        return lines
    for row in rows:
        lines.append(
            f"{row['name'][:22]:<22} "
            f"{row['start']:<10} "
            f"{row['end']:<10} "
            f"{row['total']:>7.2f} "
            f"{row['cagr']:>7.2f} "
            f"{row['maxdd']:>7.2f} "
            f"{row['sharpe']:>7.2f} "
            f"{row['final_eq']:>8.2f}"
        )
    return lines


def _momentum_scores(
    prices: pd.DataFrame,
    strategy_config: AllocatorStrategyConfig,
) -> Dict[str, float]:
    scores = {}
    for asset in strategy_config.risk_assets:
        series = prices[asset]
        ret_90 = series.iloc[-1] / series.iloc[-(MOMENTUM_FAST_DAYS + 1)] - 1
        ret_180 = series.iloc[-1] / series.iloc[-(MOMENTUM_SLOW_DAYS + 1)] - 1
        score = float(0.6 * ret_90 + 0.4 * ret_180)
        if strategy_config.score_mode == "vol_adjusted":
            vol = float(series.pct_change().iloc[-VOL_LOOKBACK_DAYS:].std(ddof=0))
            if vol > 0:
                score /= vol
        scores[asset] = score
    return scores


def _is_risk_on(
    prices: pd.DataFrame,
    strategy_config: AllocatorStrategyConfig,
) -> bool:
    if strategy_config.risk_mode == "always_on":
        return True
    if strategy_config.risk_mode == "spy_qqq_200":
        return _above_sma(prices, "SPY") and _above_sma(prices, "QQQ")
    if strategy_config.risk_mode == "breadth_2":
        return (
            sum(1 for asset in strategy_config.risk_assets if _above_sma(prices, asset))
            >= 2
        )
    if strategy_config.risk_mode == "breadth_3":
        return (
            sum(1 for asset in strategy_config.risk_assets if _above_sma(prices, asset))
            >= 3
        )
    return _above_sma(prices, "SPY")


def _is_macro_stressed(
    prices: pd.DataFrame,
    strategy_config: AllocatorStrategyConfig,
) -> bool:
    if strategy_config.macro_stress_mode not in ("vix", "tlt", "vix_tlt"):
        return False

    if (
        strategy_config.macro_stress_mode in ("vix", "vix_tlt")
        and "VIX" in prices.columns
        and len(prices) >= 20
    ):
        vix = float(prices["VIX"].iloc[-1])
        vix_sma = float(prices["VIX"].iloc[-20:].mean())
        if vix >= VIX_STRESS_LEVEL:
            return True
        if vix_sma > 0 and vix / vix_sma - 1 >= VIX_STRESS_RATIO:
            return True

    if (
        strategy_config.macro_stress_mode in ("tlt", "vix_tlt")
        and "TLT" in prices.columns
        and len(prices) > TLT_STRESS_LOOKBACK_DAYS
    ):
        tlt_return = (
            float(prices["TLT"].iloc[-1])
            / float(prices["TLT"].iloc[-(TLT_STRESS_LOOKBACK_DAYS + 1)])
            - 1
        )
        if tlt_return <= TLT_STRESS_RETURN and not _above_sma(prices, "TLT"):
            return True

    return False


def _is_asset_eligible(
    prices: pd.DataFrame,
    asset: str,
    strategy_config: AllocatorStrategyConfig,
) -> bool:
    if not strategy_config.asset_trend_filter:
        return True
    return _above_sma(prices, asset)


def _build_target_allocation(
    positive: List[Tuple[str, float]],
    strategy_config: AllocatorStrategyConfig,
) -> Dict[str, float]:
    allocation = _cash_allocation()
    selected = positive[:2]
    if selected and strategy_config.weighting == "top1":
        allocation[selected[0][0]] = 1.0
        allocation["cash"] = 0.0
    elif len(selected) >= 2:
        if strategy_config.weighting == "score_weighted":
            total_score = sum(score for _, score in selected)
            for asset, score in selected:
                allocation[asset] = score / total_score if total_score > 0 else 0.0
        else:
            allocation[selected[0][0]] = 0.5
            allocation[selected[1][0]] = 0.5
        allocation["cash"] = 0.0
    elif len(selected) == 1:
        allocation[selected[0][0]] = 0.6
        allocation["cash"] = 0.4
    return _apply_allocation_caps(allocation, strategy_config)


def _defensive_allocation(
    prices: pd.DataFrame,
    strategy_config: AllocatorStrategyConfig,
) -> Dict[str, float]:
    if strategy_config.defensive_mode == "cash":
        return _cash_allocation()

    available = [asset for asset in strategy_config.defensive_assets if asset in prices.columns]
    if not available:
        return _cash_allocation()

    scores = _defensive_scores(prices, available)
    positive = [
        (asset, score)
        for asset, score in scores.items()
        if score == score and score > 0 and _above_sma(prices, asset)
    ]
    positive.sort(key=lambda item: item[1], reverse=True)
    if not positive:
        return _cash_allocation()

    allocation = _cash_allocation()
    if strategy_config.defensive_mode == "top2_equal" and len(positive) >= 2:
        allocation[positive[0][0]] = 0.5
        allocation[positive[1][0]] = 0.5
        allocation["cash"] = 0.0
    else:
        allocation[positive[0][0]] = 1.0
        allocation["cash"] = 0.0
    return allocation


def _defensive_scores(
    prices: pd.DataFrame,
    assets: List[str],
) -> Dict[str, float]:
    scores = {}
    for asset in assets:
        series = prices[asset]
        ret_90 = series.iloc[-1] / series.iloc[-(MOMENTUM_FAST_DAYS + 1)] - 1
        ret_180 = series.iloc[-1] / series.iloc[-(MOMENTUM_SLOW_DAYS + 1)] - 1
        scores[asset] = float(0.6 * ret_90 + 0.4 * ret_180)
    return scores


def _apply_allocation_caps(
    allocation: Dict[str, float],
    strategy_config: AllocatorStrategyConfig,
) -> Dict[str, float]:
    capped = allocation.copy()
    if strategy_config.max_single_asset is not None:
        for asset in strategy_config.risk_assets:
            capped[asset] = min(capped[asset], strategy_config.max_single_asset)

    if strategy_config.max_crypto_weight is not None:
        crypto_weight = sum(capped[asset] for asset in CRYPTO_ASSETS)
        if crypto_weight > strategy_config.max_crypto_weight and crypto_weight > 0:
            scale = strategy_config.max_crypto_weight / crypto_weight
            for asset in CRYPTO_ASSETS:
                capped[asset] *= scale

    risky_weight = sum(capped[asset] for asset in TRADABLE_ASSETS)
    capped["cash"] = max(0.0, 1.0 - risky_weight)
    return capped


def _above_sma(prices: pd.DataFrame, asset: str) -> bool:
    close = float(prices[asset].iloc[-1])
    sma = float(prices[asset].iloc[-SPY_SMA_DAYS:].mean())
    return close > sma


def _oos_rows(results: Dict[str, PortfolioResult]) -> List[Dict[str, float]]:
    allocator = results["Allocator"]
    rows = []
    index_positions = np.array_split(np.arange(len(allocator.equity_curve)), 4)
    for chunk_number, positions in enumerate(index_positions, start=1):
        if len(positions) == 0:
            continue
        start_pos = int(positions[0])
        end_pos = int(positions[-1])
        start_date = allocator.equity_curve.index[start_pos]
        end_date = allocator.equity_curve.index[end_pos]
        row = {
            "label": f"{chunk_number}: {start_date.date()}->{end_date.date()}",
            "days": len(positions),
        }
        for name, result in results.items():
            row[name] = _segment_return_pct(result, start_pos, end_pos)
        row["Allocator_maxdd"] = _segment_max_drawdown_pct(allocator, start_pos, end_pos)
        rows.append(row)
    return rows


def _segment_return_pct(result: PortfolioResult, start_pos: int, end_pos: int) -> float:
    start_equity = (
        result.initial_capital
        if start_pos == 0
        else float(result.equity_curve.iloc[start_pos - 1])
    )
    end_equity = float(result.equity_curve.iloc[end_pos])
    return (end_equity / start_equity - 1) * 100


def _segment_max_drawdown_pct(
    result: PortfolioResult,
    start_pos: int,
    end_pos: int,
) -> float:
    start_equity = (
        result.initial_capital
        if start_pos == 0
        else float(result.equity_curve.iloc[start_pos - 1])
    )
    values = np.array(
        [start_equity]
        + result.equity_curve.iloc[start_pos : end_pos + 1].astype(float).tolist()
    )
    peaks = np.maximum.accumulate(values)
    return float((values / peaks - 1).min() * 100)


def _date_window(
    prices: pd.DataFrame,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> Tuple[int, int]:
    dates = prices.index
    start_ts = pd.Timestamp(start_date).normalize() if start_date is not None else dates[1]
    end_ts = pd.Timestamp(end_date).normalize() if end_date is not None else dates[-1]
    start_i = int(np.searchsorted(dates, start_ts, side="left"))
    end_i = int(np.searchsorted(dates, end_ts, side="right") - 1)
    start_i = max(1, start_i)
    if start_i > end_i:
        raise ValueError("No price rows in requested allocator backtest window")
    return start_i, end_i


def _clean_prices(
    prices: pd.DataFrame,
    required_assets: Tuple[str, ...] = RISK_ASSETS,
    asset_scope: Tuple[str, ...] = PRICE_ASSETS,
) -> pd.DataFrame:
    missing = [asset for asset in required_assets if asset not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")
    columns = [asset for asset in asset_scope if asset in prices.columns]
    cleaned = prices.loc[:, columns].copy()
    cleaned.index = pd.to_datetime(cleaned.index).normalize()
    cleaned = cleaned.sort_index().dropna()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned


def _strategy_price_assets(
    strategy_config: AllocatorStrategyConfig,
) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            strategy_config.risk_assets
            + strategy_config.defensive_assets
            + strategy_config.macro_assets
        )
    )


def _cash_allocation() -> Dict[str, float]:
    return {asset: 0.0 for asset in TRADABLE_ASSETS} | {"cash": 1.0}


def _risky_turnover(
    current: Dict[str, float],
    target: Dict[str, float],
) -> float:
    return sum(abs(target[asset] - current[asset]) for asset in TRADABLE_ASSETS)


def _priced_assets(prices: pd.DataFrame) -> List[str]:
    return [asset for asset in TRADABLE_ASSETS if asset in prices.columns]


def _should_rebalance(
    previous: pd.Timestamp,
    current: pd.Timestamp,
    strategy_config: AllocatorStrategyConfig,
) -> bool:
    if strategy_config.rebalance == "monthly":
        return (previous.year, previous.month) != (current.year, current.month)
    return _is_week_changed(previous, current)


def _oos_summary(result: PortfolioResult) -> Tuple[int, float]:
    positives = 0
    worst = 0.0
    index_positions = np.array_split(np.arange(len(result.equity_curve)), 4)
    for positions in index_positions:
        if len(positions) == 0:
            continue
        segment_return = _segment_return_pct(result, int(positions[0]), int(positions[-1]))
        if segment_return > 0:
            positives += 1
        worst = min(worst, segment_return)
    return positives, worst


def _is_week_changed(previous: pd.Timestamp, current: pd.Timestamp) -> bool:
    prev_key = previous.isocalendar()[:2]
    current_key = current.isocalendar()[:2]
    return prev_key != current_key


def _asset_week_counts(allocations: pd.DataFrame) -> Dict[str, int]:
    counts = {asset: 0 for asset in ALL_ASSETS}
    if allocations.empty:
        return counts
    week_keys = pd.Series(
        [date.isocalendar()[:2] for date in allocations.index],
        index=allocations.index,
    )
    for _, group in allocations.groupby(week_keys):
        for asset in ALL_ASSETS:
            if (group[asset] > 0).any():
                counts[asset] += 1
    return counts


def _unix_seconds(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())
