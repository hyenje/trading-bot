"""Prospective, research-only observer for public source signals.

The observer is deliberately isolated from allocation and order execution.  Its
SQLite ledger is append-oriented: baseline posts are excluded, newly admitted
events keep their original classification, and each 5/20/60-session outcome is
written once after it becomes observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from research.market_data import fetch_yahoo_adjusted_close
from research.source_signal_backtest import (
    DEFAULT_BENCHMARKS,
    DEFAULT_HORIZONS,
    HOUSE_COOLDOWN_DAYS,
    ROUND_TRIP_COST_BPS,
    SOURCES,
    SOURCE_COOLDOWN_DAYS,
    build_events,
    collect_snapshots,
    snapshots_to_posts,
)


DEFAULT_OUTPUT_DIR = Path("output/source-signals")
DEFAULT_DB_NAME = "forward_observer.sqlite3"
CLASSIFIER_VERSION = "strict-ticker-local-v1"
SOURCE_UNIVERSE_VERSION = "semiconductor-ai-sources-v1-2026-07-21"
CALCULATION_VERSION = "paired-adjusted-close-next-session-v1"
PRICE_PROVIDER = "yahoo-chart-adjusted-close"
BACKFILL_MAX_AGE_HOURS = 48
NEW_YORK = ZoneInfo("America/New_York")


SCHEMA = """
CREATE TABLE IF NOT EXISTS source_state (
    handle TEXT PRIMARY KEY,
    baseline_completed_utc TEXT NOT NULL,
    last_successful_poll_utc TEXT NOT NULL,
    last_snapshot_posts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    handle TEXT NOT NULL,
    tweet_id TEXT NOT NULL,
    post_created_at_utc TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    source_universe_version TEXT NOT NULL,
    PRIMARY KEY (handle, tweet_id)
);

CREATE TABLE IF NOT EXISTS post_amendments (
    handle TEXT NOT NULL,
    tweet_id TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    PRIMARY KEY (handle, tweet_id, text_sha256)
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    handle TEXT NOT NULL,
    house TEXT NOT NULL,
    tweet_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    direction_reason TEXT NOT NULL,
    post_created_at_utc TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    classifier_version TEXT NOT NULL,
    source_universe_version TEXT NOT NULL,
    status TEXT NOT NULL,
    admitted INTEGER NOT NULL CHECK (admitted IN (0, 1)),
    source_independent INTEGER NOT NULL CHECK (source_independent IN (0, 1)),
    house_independent INTEGER NOT NULL CHECK (house_independent IN (0, 1)),
    exclusion_reason TEXT,
    entry_date TEXT,
    entry_price REAL,
    soxx_entry_price REAL,
    qqq_entry_price REAL,
    entry_price_provider TEXT,
    entry_price_fetched_at_utc TEXT,
    data_error TEXT,
    UNIQUE (handle, tweet_id, ticker),
    FOREIGN KEY (handle, tweet_id) REFERENCES posts(handle, tweet_id)
);

CREATE TABLE IF NOT EXISTS horizons (
    event_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    status TEXT NOT NULL,
    exit_date TEXT,
    stock_entry_price REAL,
    stock_exit_price REAL,
    soxx_entry_price REAL,
    soxx_exit_price REAL,
    qqq_entry_price REAL,
    qqq_exit_price REAL,
    stock_return REAL,
    soxx_return REAL,
    qqq_return REAL,
    net_soxx_alpha REAL,
    net_qqq_alpha REAL,
    net_signal_return REAL,
    price_provider TEXT,
    price_fetched_at_utc TEXT,
    calculation_version TEXT NOT NULL,
    last_error TEXT,
    PRIMARY KEY (event_id, horizon),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
"""


def open_ledger(path: Path) -> sqlite3.Connection:
    """Open and initialize the prospective ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    _migrate_legacy_ledger(connection)
    return connection


def _migrate_legacy_ledger(connection: sqlite3.Connection) -> None:
    """Add v1 columns missing from ledgers created by an earlier local draft."""
    existing = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(horizons)")
    }
    with connection:
        for column in ("stock_entry_price", "soxx_entry_price", "qqq_entry_price"):
            if column not in existing:
                connection.execute(f"ALTER TABLE horizons ADD COLUMN {column} REAL")


def ingest_snapshots(
    connection: sqlite3.Connection,
    snapshots: Sequence[Mapping[str, object]],
    observed_at_utc: pd.Timestamp,
    failed_handles: Iterable[str] = (),
) -> Dict[str, int]:
    """Append unseen posts/events and advance only successful source checkpoints."""
    observed_at = _as_utc(observed_at_utc)
    failed = set(failed_handles)
    stats = {
        "successful_sources": 0,
        "failed_sources": len(failed),
        "new_posts": 0,
        "baseline_events": 0,
        "backfill_events": 0,
        "cooldown_events": 0,
        "admitted_events": 0,
        "amendments": 0,
    }
    ordered = sorted(
        snapshots,
        key=lambda snapshot: (
            _snapshot_retrieval_time(snapshot, observed_at),
            str(snapshot.get("source", {}).get("handle", "")),
        ),
    )
    for snapshot in ordered:
        source = snapshot.get("source", {})
        if not isinstance(source, Mapping):
            continue
        handle = str(source.get("handle", ""))
        if not handle or handle in failed:
            continue
        _ingest_source_snapshot(connection, snapshot, observed_at, stats)
        stats["successful_sources"] += 1
    return stats


def _ingest_source_snapshot(
    connection: sqlite3.Connection,
    snapshot: Mapping[str, object],
    observed_at: pd.Timestamp,
    stats: Dict[str, int],
) -> None:
    source = snapshot["source"]
    assert isinstance(source, Mapping)
    handle = str(source["handle"])
    previous_state = connection.execute(
        "SELECT * FROM source_state WHERE handle = ?", (handle,)
    ).fetchone()
    is_baseline = previous_state is None
    previous_poll = (
        _as_utc(previous_state["last_successful_poll_utc"])
        if previous_state is not None
        else None
    )
    prospective_start = (
        _as_utc(previous_state["baseline_completed_utc"])
        if previous_state is not None
        else None
    )
    posts = snapshots_to_posts([snapshot])
    candidate_events = build_events(posts)
    candidate_by_post: Dict[str, list[Mapping[str, object]]] = {}
    for event in candidate_events.to_dict("records"):
        candidate_by_post.setdefault(str(event["tweet_id"]), []).append(event)

    retrieval_time = _snapshot_retrieval_time(snapshot, observed_at)
    with connection:
        for post in posts.to_dict("records"):
            tweet_id = str(post["tweet_id"])
            text = str(post["text"])
            text_sha256 = _text_hash(text)
            existing = connection.execute(
                "SELECT text_sha256 FROM posts WHERE handle = ? AND tweet_id = ?",
                (handle, tweet_id),
            ).fetchone()
            if existing is not None:
                if existing["text_sha256"] != text_sha256:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO post_amendments
                            (handle, tweet_id, observed_at_utc, text, text_sha256)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (handle, tweet_id, _utc_iso(retrieval_time), text, text_sha256),
                    )
                    stats["amendments"] += int(cursor.rowcount > 0)
                continue

            created_at = _as_utc(post["created_at_utc"])
            connection.execute(
                """
                INSERT INTO posts
                    (handle, tweet_id, post_created_at_utc, first_seen_utc, text,
                     text_sha256, source_universe_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    handle,
                    tweet_id,
                    _utc_iso(created_at),
                    _utc_iso(retrieval_time),
                    text,
                    text_sha256,
                    SOURCE_UNIVERSE_VERSION,
                ),
            )
            stats["new_posts"] += 1
            for event in sorted(
                candidate_by_post.get(tweet_id, []), key=lambda row: str(row["ticker"])
            ):
                _insert_event(
                    connection,
                    event,
                    created_at=created_at,
                    first_seen=retrieval_time,
                    text_sha256=text_sha256,
                    is_baseline=is_baseline,
                    previous_poll=previous_poll,
                    prospective_start=prospective_start,
                    stats=stats,
                )

        baseline_completed = (
            _utc_iso(observed_at)
            if is_baseline
            else str(previous_state["baseline_completed_utc"])
        )
        checkpoint = retrieval_time
        if previous_poll is not None:
            checkpoint = max(previous_poll, retrieval_time)
        connection.execute(
            """
            INSERT INTO source_state
                (handle, baseline_completed_utc, last_successful_poll_utc,
                 last_snapshot_posts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(handle) DO UPDATE SET
                last_successful_poll_utc = excluded.last_successful_poll_utc,
                last_snapshot_posts = excluded.last_snapshot_posts
            """,
            (
                handle,
                baseline_completed,
                _utc_iso(checkpoint),
                len(snapshot.get("posts", [])),
            ),
        )


def _insert_event(
    connection: sqlite3.Connection,
    event: Mapping[str, object],
    *,
    created_at: pd.Timestamp,
    first_seen: pd.Timestamp,
    text_sha256: str,
    is_baseline: bool,
    previous_poll: Optional[pd.Timestamp],
    prospective_start: Optional[pd.Timestamp],
    stats: Dict[str, int],
) -> None:
    handle = str(event["handle"])
    house = str(event["house"])
    tweet_id = str(event["tweet_id"])
    ticker = str(event["ticker"])
    direction = int(event["direction"])
    source_independent = True
    house_independent = True
    admitted = False
    exclusion_reason: Optional[str]

    if is_baseline:
        status = "EXCLUDED_BASELINE"
        exclusion_reason = "present_at_source_baseline"
        stats["baseline_events"] += 1
    elif prospective_start is not None and created_at <= prospective_start:
        status = "EXCLUDED_BACKFILL"
        exclusion_reason = "post_predates_prospective_start"
        stats["backfill_events"] += 1
    elif previous_poll is not None and created_at <= previous_poll:
        status = "EXCLUDED_BACKFILL"
        exclusion_reason = "post_predates_last_successful_poll"
        stats["backfill_events"] += 1
    elif first_seen - created_at > pd.Timedelta(hours=BACKFILL_MAX_AGE_HOURS):
        status = "EXCLUDED_BACKFILL"
        exclusion_reason = f"first_seen_more_than_{BACKFILL_MAX_AGE_HOURS}h_after_post"
        stats["backfill_events"] += 1
    else:
        source_independent = not _has_recent_admitted_event(
            connection,
            first_seen,
            SOURCE_COOLDOWN_DAYS,
            handle=handle,
            ticker=ticker,
            direction=direction,
        )
        house_independent = not _has_recent_admitted_event(
            connection,
            first_seen,
            HOUSE_COOLDOWN_DAYS,
            house=house,
            ticker=ticker,
            direction=direction,
        )
        if source_independent and house_independent:
            status = "PENDING_ENTRY"
            exclusion_reason = None
            admitted = True
            stats["admitted_events"] += 1
        else:
            status = "EXCLUDED_COOLDOWN"
            failed_checks = []
            if not source_independent:
                failed_checks.append("source_30d")
            if not house_independent:
                failed_checks.append("house_7d")
            exclusion_reason = ",".join(failed_checks)
            stats["cooldown_events"] += 1

    event_id = hashlib.sha256(
        f"{handle}|{tweet_id}|{ticker}".encode("utf-8")
    ).hexdigest()
    connection.execute(
        """
        INSERT INTO events
            (event_id, handle, house, tweet_id, ticker, direction,
             direction_reason, post_created_at_utc, first_seen_utc, text_sha256,
             classifier_version, source_universe_version, status, admitted,
             source_independent, house_independent, exclusion_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            handle,
            house,
            tweet_id,
            ticker,
            direction,
            str(event["direction_reason"]),
            _utc_iso(created_at),
            _utc_iso(first_seen),
            text_sha256,
            CLASSIFIER_VERSION,
            SOURCE_UNIVERSE_VERSION,
            status,
            int(admitted),
            int(source_independent),
            int(house_independent),
            exclusion_reason,
        ),
    )
    if admitted:
        connection.executemany(
            """
            INSERT INTO horizons
                (event_id, horizon, status, calculation_version)
            VALUES (?, ?, 'PENDING', ?)
            """,
            [
                (event_id, int(horizon), CALCULATION_VERSION)
                for horizon in DEFAULT_HORIZONS
            ],
        )


def _has_recent_admitted_event(
    connection: sqlite3.Connection,
    first_seen: pd.Timestamp,
    cooldown_days: int,
    **fields: object,
) -> bool:
    clauses = ["admitted = 1"]
    parameters: list[object] = []
    for name, value in fields.items():
        clauses.append(f"{name} = ?")
        parameters.append(value)
    clauses.append("first_seen_utc > ?")
    parameters.append(_utc_iso(first_seen - pd.Timedelta(days=cooldown_days)))
    clauses.append("first_seen_utc <= ?")
    parameters.append(_utc_iso(first_seen))
    query = "SELECT 1 FROM events WHERE " + " AND ".join(clauses) + " LIMIT 1"
    return connection.execute(query, parameters).fetchone() is not None


def fetch_open_event_prices(
    connection: sqlite3.Connection,
    as_of_utc: pd.Timestamp,
) -> Tuple[Dict[str, pd.Series], list[Dict[str, str]]]:
    """Fetch only the symbols needed by admitted, incomplete events."""
    rows = connection.execute(
        """
        SELECT ticker, first_seen_utc
        FROM events
        WHERE admitted = 1 AND status != 'COMPLETE'
        """
    ).fetchall()
    if not rows:
        return {}, []
    first_seen = min(_as_utc(row["first_seen_utc"]) for row in rows)
    start = first_seen.tz_convert(NEW_YORK).tz_localize(None).normalize()
    start -= pd.Timedelta(days=10)
    end = _as_utc(as_of_utc).tz_localize(None).normalize() + pd.Timedelta(days=2)
    symbols = sorted({str(row["ticker"]) for row in rows} | set(DEFAULT_BENCHMARKS))
    prices: Dict[str, pd.Series] = {}
    failures: list[Dict[str, str]] = []
    for symbol in symbols:
        try:
            prices[symbol] = fetch_yahoo_adjusted_close(symbol, start, end)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            failures.append({"ticker": symbol, "error": str(exc)})
    return prices, failures


def evaluate_open_events(
    connection: sqlite3.Connection,
    prices: Mapping[str, pd.Series],
    as_of_utc: pd.Timestamp,
    *,
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> Dict[str, int]:
    """Advance admitted events using only fully observable daily closes."""
    as_of = _as_utc(as_of_utc)
    cutoff = latest_complete_session_date(as_of)
    normalized = {
        symbol: _normalize_prices(series, cutoff) for symbol, series in prices.items()
    }
    events = connection.execute(
        """
        SELECT * FROM events
        WHERE admitted = 1 AND status != 'COMPLETE'
        ORDER BY first_seen_utc, event_id
        """
    ).fetchall()
    stats = {"entries_opened": 0, "horizons_matured": 0, "data_errors": 0}
    for event in events:
        required = (str(event["ticker"]), *DEFAULT_BENCHMARKS)
        missing = [symbol for symbol in required if symbol not in normalized]
        if missing:
            with connection:
                _mark_data_error(
                    connection,
                    str(event["event_id"]),
                    "missing_price_series:" + ",".join(missing),
                )
            stats["data_errors"] += 1
            continue
        event_stats = _evaluate_event(
            connection,
            event,
            normalized,
            as_of,
            cost_bps=cost_bps,
        )
        for key, value in event_stats.items():
            stats[key] += value
    return stats


def _evaluate_event(
    connection: sqlite3.Connection,
    event: sqlite3.Row,
    prices: Mapping[str, pd.Series],
    as_of: pd.Timestamp,
    *,
    cost_bps: float,
) -> Dict[str, int]:
    event_id = str(event["event_id"])
    stock = prices[str(event["ticker"])]
    soxx = prices["SOXX"]
    qqq = prices["QQQ"]
    stats = {"entries_opened": 0, "horizons_matured": 0, "data_errors": 0}
    with connection:
        if event["entry_date"] is None:
            first_seen_date = (
                _as_utc(event["first_seen_utc"])
                .tz_convert(NEW_YORK)
                .tz_localize(None)
                .normalize()
            )
            candidates = stock.index[stock.index > first_seen_date]
            if candidates.empty:
                benchmark_candidates = soxx.index.intersection(qqq.index)
                benchmark_candidates = benchmark_candidates[
                    benchmark_candidates > first_seen_date
                ]
                if len(benchmark_candidates):
                    _mark_data_error(
                        connection,
                        event_id,
                        "missing_stock_entry_after_due_session:"
                        + benchmark_candidates[0].date().isoformat(),
                    )
                    stats["data_errors"] += 1
                else:
                    _set_waiting_status(connection, event_id, entry_open=False)
                return stats
            entry_date = pd.Timestamp(candidates[0])
            if entry_date not in soxx.index or entry_date not in qqq.index:
                _mark_data_error(
                    connection,
                    event_id,
                    f"benchmark_missing_on_entry:{entry_date.date().isoformat()}",
                )
                stats["data_errors"] += 1
                return stats
            connection.execute(
                """
                UPDATE events
                SET status = 'OPEN', entry_date = ?, entry_price = ?,
                    soxx_entry_price = ?, qqq_entry_price = ?,
                    entry_price_provider = ?, entry_price_fetched_at_utc = ?,
                    data_error = NULL
                WHERE event_id = ? AND entry_date IS NULL
                """,
                (
                    entry_date.date().isoformat(),
                    float(stock.loc[entry_date]),
                    float(soxx.loc[entry_date]),
                    float(qqq.loc[entry_date]),
                    PRICE_PROVIDER,
                    _utc_iso(as_of),
                    event_id,
                ),
            )
            stats["entries_opened"] += 1
            event = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            assert event is not None

        entry_date = pd.Timestamp(event["entry_date"])
        missing_entry = [
            symbol
            for symbol, series in (
                (str(event["ticker"]), stock),
                ("SOXX", soxx),
                ("QQQ", qqq),
            )
            if entry_date not in series.index
        ]
        if missing_entry:
            _mark_data_error(
                connection,
                event_id,
                "stored_entry_date_missing_from_series:" + ",".join(missing_entry),
            )
            stats["data_errors"] += 1
            return stats
        entry_position = int(stock.index.get_loc(entry_date))
        pending = connection.execute(
            """
            SELECT * FROM horizons
            WHERE event_id = ? AND status = 'PENDING'
            ORDER BY horizon
            """,
            (event_id,),
        ).fetchall()
        connection.execute(
            "UPDATE horizons SET last_error = NULL WHERE event_id = ? AND status = 'PENDING'",
            (event_id,),
        )
        errors: list[str] = []
        benchmark_sessions = soxx.index.intersection(qqq.index)
        benchmark_sessions = benchmark_sessions[benchmark_sessions > entry_date]
        for horizon_row in pending:
            horizon = int(horizon_row["horizon"])
            exit_position = entry_position + horizon
            if exit_position >= len(stock):
                if len(benchmark_sessions) >= horizon:
                    error = f"stock_horizon_overdue_{horizon}d"
                    connection.execute(
                        "UPDATE horizons SET last_error = ? WHERE event_id = ? AND horizon = ?",
                        (error, event_id, horizon),
                    )
                    errors.append(error)
                continue
            exit_date = pd.Timestamp(stock.index[exit_position])
            if exit_date not in soxx.index or exit_date not in qqq.index:
                error = f"benchmark_missing_on_exit_{horizon}d:{exit_date.date().isoformat()}"
                connection.execute(
                    "UPDATE horizons SET last_error = ? WHERE event_id = ? AND horizon = ?",
                    (error, event_id, horizon),
                )
                errors.append(error)
                continue
            stock_entry_price = float(stock.loc[entry_date])
            soxx_entry_price = float(soxx.loc[entry_date])
            qqq_entry_price = float(qqq.loc[entry_date])
            stock_return = float(stock.loc[exit_date] / stock_entry_price - 1.0)
            soxx_return = float(soxx.loc[exit_date] / soxx_entry_price - 1.0)
            qqq_return = float(qqq.loc[exit_date] / qqq_entry_price - 1.0)
            direction = int(event["direction"])
            cost_rate = cost_bps / 10_000.0
            connection.execute(
                """
                UPDATE horizons
                SET status = 'MATURED', exit_date = ?, stock_entry_price = ?,
                    stock_exit_price = ?, soxx_entry_price = ?, soxx_exit_price = ?,
                    qqq_entry_price = ?, qqq_exit_price = ?, stock_return = ?,
                    soxx_return = ?, qqq_return = ?, net_soxx_alpha = ?,
                    net_qqq_alpha = ?, net_signal_return = ?, price_provider = ?,
                    price_fetched_at_utc = ?, last_error = NULL
                WHERE event_id = ? AND horizon = ? AND status = 'PENDING'
                """,
                (
                    exit_date.date().isoformat(),
                    stock_entry_price,
                    float(stock.loc[exit_date]),
                    soxx_entry_price,
                    float(soxx.loc[exit_date]),
                    qqq_entry_price,
                    float(qqq.loc[exit_date]),
                    stock_return,
                    soxx_return,
                    qqq_return,
                    direction * (stock_return - soxx_return) - cost_rate,
                    direction * (stock_return - qqq_return) - cost_rate,
                    direction * stock_return - cost_rate,
                    PRICE_PROVIDER,
                    _utc_iso(as_of),
                    event_id,
                    horizon,
                ),
            )
            stats["horizons_matured"] += 1

        remaining = connection.execute(
            "SELECT COUNT(*) FROM horizons WHERE event_id = ? AND status = 'PENDING'",
            (event_id,),
        ).fetchone()[0]
        if errors:
            connection.execute(
                "UPDATE events SET status = 'DATA_ERROR', data_error = ? WHERE event_id = ?",
                (";".join(errors), event_id),
            )
            stats["data_errors"] += 1
        else:
            connection.execute(
                "UPDATE events SET status = ?, data_error = NULL WHERE event_id = ?",
                ("COMPLETE" if remaining == 0 else "OPEN", event_id),
            )
    return stats


def _set_waiting_status(
    connection: sqlite3.Connection, event_id: str, *, entry_open: bool
) -> None:
    connection.execute(
        "UPDATE events SET status = ?, data_error = NULL WHERE event_id = ?",
        ("OPEN" if entry_open else "PENDING_ENTRY", event_id),
    )
    connection.execute(
        "UPDATE horizons SET last_error = NULL WHERE event_id = ? AND status = 'PENDING'",
        (event_id,),
    )


def _mark_data_error(
    connection: sqlite3.Connection, event_id: str, message: str
) -> None:
    connection.execute(
        "UPDATE events SET status = 'DATA_ERROR', data_error = ? WHERE event_id = ?",
        (message, event_id),
    )
    connection.execute(
        """
        UPDATE horizons SET last_error = ?
        WHERE event_id = ? AND status = 'PENDING'
        """,
        (message, event_id),
    )


def latest_complete_session_date(as_of_utc: pd.Timestamp) -> pd.Timestamp:
    """Return the prior New York calendar date for conservative daily-bar use."""
    local = _as_utc(as_of_utc).tz_convert(NEW_YORK)
    return local.tz_localize(None).normalize() - pd.Timedelta(days=1)


def _normalize_prices(series: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    normalized = series.dropna().astype(float).copy()
    index = pd.to_datetime(normalized.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    normalized.index = index.normalize()
    normalized = normalized.groupby(normalized.index).last().sort_index()
    return normalized[normalized.index <= cutoff]


def build_forward_gate(connection: sqlite3.Connection) -> Dict[str, object]:
    """Evaluate the frozen v1 gate without ever enabling production execution."""
    admitted_total = int(
        connection.execute("SELECT COUNT(*) FROM events WHERE admitted = 1").fetchone()[0]
    )
    admitted_posts_total = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT handle, tweet_id FROM events
                WHERE admitted = 1 GROUP BY handle, tweet_id
            )
            """
        ).fetchone()[0]
    )
    data_error_events = int(
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE admitted = 1 AND status = 'DATA_ERROR'"
        ).fetchone()[0]
    )
    overdue_price_events = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE admitted = 1 AND (
                data_error LIKE '%missing_stock_entry_after_due_session%'
                OR data_error LIKE '%stock_horizon_overdue_%'
            )
            """
        ).fetchone()[0]
    )
    rows = connection.execute(
        """
        SELECT e.handle, e.tweet_id, e.ticker, e.first_seen_utc,
               h.net_soxx_alpha
        FROM events e
        JOIN horizons h ON h.event_id = e.event_id
        WHERE e.admitted = 1 AND h.horizon = 20 AND h.status = 'MATURED'
        ORDER BY e.first_seen_utc, e.event_id
        """
    ).fetchall()
    tickers = [str(row["ticker"]) for row in rows]
    if len(rows):
        frame = pd.DataFrame([dict(row) for row in rows])
        posts = (
            frame.groupby(["handle", "tweet_id", "first_seen_utc"], as_index=False)[
                "net_soxx_alpha"
            ]
            .mean()
        )
        posts["observation_date"] = posts["first_seen_utc"].map(
            lambda value: _as_utc(value).tz_convert(NEW_YORK).date().isoformat()
        )
        daily = posts.groupby("observation_date")["net_soxx_alpha"].mean().sort_index()
        values = daily.to_numpy(dtype=float)
        rng = np.random.default_rng(20260721)
        bootstrap = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        counts = pd.Series(tickers).value_counts(normalize=True)
        metrics = {
            "admitted_events_total": admitted_total,
            "admitted_signal_posts_total": admitted_posts_total,
            "matured_20d_events": len(rows),
            "matured_20d_signal_posts": int(len(posts)),
            "matured_20d_observation_days": int(len(daily)),
            "pending_or_data_error_20d_events": admitted_total - len(rows),
            "data_error_events": data_error_events,
            "overdue_price_events": overdue_price_events,
            "tickers": len(set(tickers)),
            "span_days": int((pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days),
            "median_daily_cluster_net_soxx_alpha": float(np.median(values)),
            "daily_cluster_win_rate": float((values > 0).mean()),
            "daily_cluster_bootstrap_mean_ci_low": float(np.quantile(bootstrap, 0.025)),
            "daily_cluster_bootstrap_mean_ci_high": float(np.quantile(bootstrap, 0.975)),
            "max_ticker_share": float(counts.max()),
        }
    else:
        metrics = {
            "admitted_events_total": admitted_total,
            "admitted_signal_posts_total": admitted_posts_total,
            "matured_20d_events": 0,
            "matured_20d_signal_posts": 0,
            "matured_20d_observation_days": 0,
            "pending_or_data_error_20d_events": admitted_total,
            "data_error_events": data_error_events,
            "overdue_price_events": overdue_price_events,
            "tickers": 0,
            "span_days": 0,
            "median_daily_cluster_net_soxx_alpha": None,
            "daily_cluster_win_rate": None,
            "daily_cluster_bootstrap_mean_ci_low": None,
            "daily_cluster_bootstrap_mean_ci_high": None,
            "max_ticker_share": None,
        }
    checks = {
        "at_least_20_matured_20d_events": metrics["matured_20d_events"] >= 20,
        "at_least_20_matured_signal_posts": metrics["matured_20d_signal_posts"] >= 20,
        "at_least_20_observation_days": metrics["matured_20d_observation_days"] >= 20,
        "at_least_4_tickers": metrics["tickers"] >= 4,
        "at_least_180_days": metrics["span_days"] >= 180,
        "positive_median_daily_cluster_net_soxx_alpha": _positive(
            metrics["median_daily_cluster_net_soxx_alpha"]
        ),
        "daily_cluster_win_rate_at_least_55pct": (
            metrics["daily_cluster_win_rate"] is not None
            and metrics["daily_cluster_win_rate"] >= 0.55
        ),
        "positive_daily_cluster_bootstrap_mean_ci": _positive(
            metrics["daily_cluster_bootstrap_mean_ci_low"]
        ),
        "max_ticker_share_at_most_35pct": (
            metrics["max_ticker_share"] is not None
            and metrics["max_ticker_share"] <= 0.35
        ),
        "no_data_error_events": metrics["data_error_events"] == 0,
        "no_overdue_price_events": metrics["overdue_price_events"] == 0,
    }
    passed = all(checks.values())
    return {
        "status": "RESEARCH_ONLY",
        "phase": "PAPER_REVIEW_ELIGIBLE" if passed else "COLLECTING_FORWARD_DATA",
        "production_signal_enabled": False,
        "checks": checks,
        "metrics": metrics,
        "decision": (
            "Review as a separate paper overlay; do not connect it to execution."
            if passed
            else "Keep collecting forward observations; do not alter allocator weights."
        ),
    }


def export_observer_outputs(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    generated_at_utc: pd.Timestamp,
    collection_mode: str,
    ingestion_stats: Mapping[str, int],
    evaluation_stats: Mapping[str, int],
    snapshot_failures: Sequence[Mapping[str, str]],
    price_failures: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    """Export human-readable mirrors; SQLite remains the source of truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.read_sql_query(
        "SELECT * FROM source_state ORDER BY handle", connection
    ).to_csv(output_dir / "forward_sources.csv", index=False)
    pd.read_sql_query(
        "SELECT * FROM events ORDER BY first_seen_utc, event_id", connection
    ).to_csv(output_dir / "forward_events.csv", index=False)
    pd.read_sql_query(
        "SELECT * FROM horizons ORDER BY event_id, horizon", connection
    ).to_csv(output_dir / "forward_horizons.csv", index=False)
    gate = build_forward_gate(connection)
    status_counts = {
        str(row["status"]): int(row["count"])
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM events GROUP BY status"
        )
    }
    checkpoint = connection.execute(
        """
        SELECT MIN(last_successful_poll_utc) AS first,
               MAX(last_successful_poll_utc) AS last
        FROM source_state
        """
    ).fetchone()
    manifest = {
        "generated_at_utc": _utc_iso(generated_at_utc),
        "collection_mode": collection_mode,
        "ledger": DEFAULT_DB_NAME,
        "ledger_counts": {
            "sources": int(
                connection.execute("SELECT COUNT(*) FROM source_state").fetchone()[0]
            ),
            "posts": int(connection.execute("SELECT COUNT(*) FROM posts").fetchone()[0]),
            "events": int(
                connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            ),
            "horizons": int(
                connection.execute("SELECT COUNT(*) FROM horizons").fetchone()[0]
            ),
            "event_statuses": status_counts,
        },
        "source_checkpoint_utc": {
            "earliest": checkpoint["first"],
            "latest": checkpoint["last"],
        },
        "prospective_start_rule": (
            "Each source's first successful snapshot is EXCLUDED_BASELINE."
        ),
        "method": {
            "classifier_version": CLASSIFIER_VERSION,
            "source_universe_version": SOURCE_UNIVERSE_VERSION,
            "entry": (
                "first available adjusted close on a US trading date strictly "
                "after the New York calendar date of first_seen_utc"
            ),
            "horizons_trading_sessions": list(DEFAULT_HORIZONS),
            "benchmarks": list(DEFAULT_BENCHMARKS),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "source_cooldown_days": SOURCE_COOLDOWN_DAYS,
            "house_cooldown_days": HOUSE_COOLDOWN_DAYS,
            "backfill_max_age_hours": BACKFILL_MAX_AGE_HOURS,
            "price_provider": PRICE_PROVIDER,
            "gate_inference_unit": (
                "Ticker events are averaged per source post, then per New York "
                "observation date before win-rate and bootstrap calculations."
            ),
            "adjusted_close_pairing": (
                "Each matured horizon stores entry and exit adjusted closes from "
                "the same fetch so later corporate-action revisions cannot mix scales."
            ),
        },
        "ingestion": dict(ingestion_stats),
        "evaluation": dict(evaluation_stats),
        "snapshot_failures": list(snapshot_failures),
        "price_failures": list(price_failures),
        "gate": gate,
        "limitations": [
            "The public embedded-profile feed is incomplete and selection is undocumented.",
            "The source universe and rules were selected retrospectively before v1 began.",
            "Adjusted closes can be revised after corporate actions.",
            "Short signals omit borrow availability, borrow fees, and squeeze execution.",
            "Missing prices remain pending or DATA_ERROR and are not silently discarded.",
        ],
    }
    _atomic_write_json(output_dir / "forward_observer_manifest.json", manifest)
    return manifest


def run_once(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    use_cache: bool = False,
    observed_at_utc: Optional[pd.Timestamp] = None,
) -> Dict[str, object]:
    run_started_at = _as_utc(
        pd.Timestamp.now(tz="UTC") if observed_at_utc is None else observed_at_utc
    )
    snapshots, snapshot_failures = collect_snapshots(
        output_dir,
        refresh=not use_cache,
    )
    observed_at = (
        run_started_at
        if observed_at_utc is not None
        else _as_utc(pd.Timestamp.now(tz="UTC"))
    )
    failed_handles = {failure["handle"] for failure in snapshot_failures}
    ledger_path = output_dir / DEFAULT_DB_NAME
    connection = open_ledger(ledger_path)
    try:
        ingestion = ingest_snapshots(
            connection,
            snapshots,
            observed_at,
            failed_handles=failed_handles,
        )
        prices, price_failures = fetch_open_event_prices(connection, observed_at)
        evaluation = evaluate_open_events(connection, prices, observed_at)
        return export_observer_outputs(
            connection,
            output_dir,
            generated_at_utc=observed_at,
            collection_mode="cache" if use_cache else "live",
            ingestion_stats=ingestion,
            evaluation_stats=evaluation,
            snapshot_failures=snapshot_failures,
            price_failures=price_failures,
        )
    finally:
        connection.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one poll/evaluation cycle; continuous execution is not supported.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use frozen snapshots (appropriate only for the initial baseline/replay).",
    )
    args = parser.parse_args(argv)
    if not args.once:
        parser.error("--once is required; this observer never loops or trades")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_once(args.output_dir, use_cache=args.use_cache)
    gate = manifest["gate"]
    ingestion = manifest["ingestion"]
    evaluation = manifest["evaluation"]
    print(f"Successful sources: {ingestion['successful_sources']}/{len(SOURCES)}")
    print(f"New posts: {ingestion['new_posts']}")
    print(f"Forward events admitted: {ingestion['admitted_events']}")
    print(f"Horizons matured this run: {evaluation['horizons_matured']}")
    print(f"Gate: {gate['status']} / {gate['phase']}")
    print(gate["decision"])
    return 0


def _snapshot_retrieval_time(
    snapshot: Mapping[str, object], observed_at: pd.Timestamp
) -> pd.Timestamp:
    value = snapshot.get("retrieved_at_utc")
    if value is None:
        return observed_at
    return min(_as_utc(value), observed_at)


def _as_utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _utc_iso(value: object) -> str:
    return _as_utc(value).isoformat()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _positive(value: object) -> bool:
    return value is not None and float(value) > 0


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
