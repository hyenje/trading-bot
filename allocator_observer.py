"""
Market regime allocator observation mode.

This is a read-only observer. It fetches public market data, computes the
current target allocation, and exposes it to the dashboard/status API.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict

import pandas as pd

from backtesting.market_regime_allocator import (
    DEFAULT_DAYS,
    build_current_allocator_signal,
    fetch_market_regime_prices,
    tlt_stress_candidate_config,
)


class AllocatorObserver:
    """Read-only TLT-stress allocator signal observer."""

    def __init__(
        self,
        price_loader: Callable[[int], pd.DataFrame] = fetch_market_regime_prices,
        days: int = DEFAULT_DAYS,
        cache_ttl_seconds: int = 900,
    ):
        self.price_loader = price_loader
        self.days = days
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.strategy_config = tlt_stress_candidate_config()
        self._cached_signal: Dict[str, Any] | None = None
        self._cached_at: datetime | None = None
        self._last_error = ""

    def get_status(self) -> Dict[str, Any]:
        signal = self.get_signal()
        return {
            "running": True,
            "observer_mode": True,
            "allocator_observer_mode": True,
            "dry_run": True,
            "balance": 0,
            "daily_pnl": 0,
            "open_positions": 0,
            "daily_trades": 0,
            "daily_win_rate": 0,
            "total_trades": 0,
            "positions": [],
            "recent_trades": [],
            "equity_curve": [],
            "allocator_signal": signal,
            "timestamp": datetime.now().isoformat(),
        }

    def get_signal(self) -> Dict[str, Any]:
        now = datetime.now()
        if (
            self._cached_signal is not None
            and self._cached_at is not None
            and now - self._cached_at < self.cache_ttl
        ):
            return self._cached_signal

        try:
            prices = self.price_loader(self.days)
            signal = build_current_allocator_signal(prices, self.strategy_config)
            signal["updated_at"] = now.isoformat()
            signal["cache_ttl_seconds"] = int(self.cache_ttl.total_seconds())
            signal["error"] = ""
            self._cached_signal = signal
            self._cached_at = now
            self._last_error = ""
            return signal
        except Exception as exc:
            self._last_error = str(exc)
            if self._cached_signal is not None:
                stale = dict(self._cached_signal)
                stale["stale"] = True
                stale["error"] = self._last_error
                stale["updated_at"] = self._cached_at.isoformat() if self._cached_at else ""
                return stale
            return {
                "strategy": self.strategy_config.name,
                "asof_date": "",
                "decision": "ERROR",
                "risk_on": False,
                "macro_stressed": False,
                "macro_reasons": [],
                "allocation": {},
                "nonzero_allocation": {},
                "scores": {},
                "latest_prices": {},
                "updated_at": now.isoformat(),
                "cache_ttl_seconds": int(self.cache_ttl.total_seconds()),
                "error": self._last_error,
            }


def format_allocator_signal(signal: Dict[str, Any]) -> str:
    allocation = signal.get("nonzero_allocation") or {}
    allocation_text = ", ".join(
        f"{asset}={float(weight) * 100:.0f}%"
        for asset, weight in allocation.items()
    ) or "cash=100%"
    reasons = signal.get("macro_reasons") or []
    reason_text = "; ".join(reasons) if reasons else "-"
    prices = signal.get("latest_prices") or {}
    price_text = ", ".join(
        f"{asset}={float(price):.2f}"
        for asset, price in prices.items()
        if asset in {"SPY", "QQQ", "BTC", "ETH", "TLT"}
    )

    return "\n".join(
        [
            "=== Allocator Signal ===",
            f"Strategy: {signal.get('strategy', '-')}",
            f"As of: {signal.get('asof_date', '-')}",
            f"Decision: {signal.get('decision', '-')}",
            f"Target allocation: {allocation_text}",
            f"Macro stress: {'YES' if signal.get('macro_stressed') else 'NO'}",
            f"Reason: {reason_text}",
            f"Prices: {price_text or '-'}",
            f"Updated: {signal.get('updated_at', '-')}",
            f"Error: {signal.get('error') or '-'}",
        ]
    )
