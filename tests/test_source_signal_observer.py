import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.source_signal_observer import (
    build_forward_gate,
    evaluate_open_events,
    ingest_snapshots,
    open_ledger,
)


class SourceSignalObserverTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.connection = open_ledger(Path(self.temporary.name) / "observer.sqlite3")

    def tearDown(self):
        self.connection.close()
        self.temporary.cleanup()

    def test_first_successful_snapshot_is_per_source_baseline_and_idempotent(self):
        first_seen = pd.Timestamp("2025-01-03T00:00:00Z")
        snapshot = self._snapshot(
            "source_a",
            "house_a",
            first_seen,
            [self._post("101", "2025-01-02T23:30:00Z", "Bullish $NVDA")],
        )

        ingest_snapshots(self.connection, [snapshot], first_seen)
        original = self.connection.execute("SELECT * FROM events").fetchone()
        ingest_snapshots(
            self.connection,
            [
                self._snapshot(
                    "source_a",
                    "house_a",
                    pd.Timestamp("2025-01-04T00:00:00Z"),
                    [self._post("101", "2025-01-02T23:30:00Z", "Bearish $NVDA")],
                )
            ],
            pd.Timestamp("2025-01-04T00:00:00Z"),
        )

        current = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(current["status"], "EXCLUDED_BASELINE")
        self.assertEqual(current["admitted"], 0)
        self.assertEqual(current["first_seen_utc"], original["first_seen_utc"])
        self.assertEqual(current["direction"], original["direction"])
        self.assertEqual(current["text_sha256"], original["text_sha256"])
        self.assertEqual(self._count("posts"), 1)
        self.assertEqual(self._count("events"), 1)
        self.assertEqual(self._count("horizons"), 0)
        self.assertEqual(self._count("post_amendments"), 1)
        self.assertEqual(build_forward_gate(self.connection)["metrics"]["matured_20d_events"], 0)

        ingest_snapshots(
            self.connection,
            [self._snapshot("source_b", "house_b", first_seen, [])],
            first_seen,
        )
        states = self.connection.execute(
            "SELECT handle FROM source_state ORDER BY handle"
        ).fetchall()
        self.assertEqual([row["handle"] for row in states], ["source_a", "source_b"])

    def test_late_discovery_is_backfill_but_recent_signal_is_admitted(self):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        observed = pd.Timestamp("2025-01-05T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [self._snapshot("source_a", "house_a", baseline, [])],
            baseline,
        )
        snapshot = self._snapshot(
            "source_a",
            "house_a",
            observed,
            [
                self._post(
                    "old",
                    observed - pd.Timedelta(hours=49),
                    "Bullish $NVDA",
                ),
                self._post(
                    "recent",
                    observed - pd.Timedelta(hours=47),
                    "Bullish $AMD",
                ),
            ],
        )

        ingest_snapshots(self.connection, [snapshot], observed)

        rows = {
            row["tweet_id"]: row
            for row in self.connection.execute("SELECT * FROM events").fetchall()
        }
        self.assertEqual(rows["old"]["status"], "EXCLUDED_BACKFILL")
        self.assertEqual(rows["old"]["admitted"], 0)
        self.assertEqual(rows["recent"]["status"], "PENDING_ENTRY")
        self.assertEqual(rows["recent"]["admitted"], 1)
        self.assertEqual(self._count("horizons"), 3)

    def test_posts_from_before_prospective_start_are_never_admitted(self):
        baseline = pd.Timestamp("2025-01-05T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [
                self._snapshot(
                    "source_a",
                    "house_a",
                    pd.Timestamp("2025-01-01T00:00:00Z"),
                    [],
                )
            ],
            baseline,
        )
        observed = baseline + pd.Timedelta(hours=2)
        newly_visible = self._snapshot(
            "source_a",
            "house_a",
            observed,
            [self._post("hidden", baseline - pd.Timedelta(minutes=1), "Bullish $NVDA")],
        )

        ingest_snapshots(self.connection, [newly_visible], observed)

        event = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(event["status"], "EXCLUDED_BACKFILL")
        self.assertEqual(event["exclusion_reason"], "post_predates_prospective_start")

    def test_cooldown_is_frozen_and_uses_only_admitted_events(self):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [
                self._snapshot("source_a", "house_a", baseline, []),
                self._snapshot("source_b", "house_a", baseline, []),
            ],
            baseline,
        )
        observations = [
            ("source_a", "a1", pd.Timestamp("2025-01-02T00:00:00Z")),
            ("source_b", "b1", pd.Timestamp("2025-01-06T00:00:00Z")),
            ("source_a", "a2", pd.Timestamp("2025-01-11T00:00:00Z")),
            ("source_a", "a3", pd.Timestamp("2025-02-01T00:00:00Z")),
        ]
        for handle, tweet_id, observed in observations:
            snapshot = self._snapshot(
                handle,
                "house_a",
                observed,
                [
                    self._post(
                        tweet_id,
                        observed - pd.Timedelta(minutes=10),
                        "Bullish $NVDA",
                    )
                ],
            )
            ingest_snapshots(self.connection, [snapshot], observed)

        rows = {
            row["tweet_id"]: row
            for row in self.connection.execute("SELECT * FROM events").fetchall()
        }
        self.assertEqual(rows["a1"]["status"], "PENDING_ENTRY")
        self.assertEqual(rows["b1"]["status"], "EXCLUDED_COOLDOWN")
        self.assertEqual(rows["b1"]["source_independent"], 1)
        self.assertEqual(rows["b1"]["house_independent"], 0)
        self.assertEqual(rows["a2"]["status"], "EXCLUDED_COOLDOWN")
        self.assertEqual(rows["a3"]["status"], "PENDING_ENTRY")
        self.assertEqual(sum(row["admitted"] for row in rows.values()), 2)

    def test_same_batch_house_cooldown_uses_retrieval_order(self):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [
                self._snapshot("aaa_late", "house_a", baseline, []),
                self._snapshot("zzz_early", "house_a", baseline, []),
            ],
            baseline,
        )
        early = pd.Timestamp("2025-01-02T00:00:00Z")
        late = early + pd.Timedelta(minutes=10)
        snapshots = [
            self._snapshot(
                "aaa_late",
                "house_a",
                late,
                [self._post("late", early, "Bullish $NVDA")],
            ),
            self._snapshot(
                "zzz_early",
                "house_a",
                early,
                [self._post("early", early, "Bullish $NVDA")],
            ),
        ]

        ingest_snapshots(
            self.connection,
            snapshots,
            late + pd.Timedelta(minutes=1),
        )

        rows = {
            row["tweet_id"]: row
            for row in self.connection.execute("SELECT * FROM events").fetchall()
        }
        self.assertEqual(rows["early"]["status"], "PENDING_ENTRY")
        self.assertEqual(rows["late"]["status"], "EXCLUDED_COOLDOWN")

    def test_entry_uses_new_york_first_seen_date(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        sessions = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
        prices = {
            "NVDA": pd.Series([99.0, 100.0, 101.0], index=sessions),
            "SOXX": pd.Series(100.0, index=sessions),
            "QQQ": pd.Series(100.0, index=sessions),
        }

        evaluate_open_events(
            self.connection,
            prices,
            pd.Timestamp("2025-01-07T18:00:00Z"),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(event["entry_date"], "2025-01-03")
        self.assertEqual(event["entry_price"], 100.0)

    def test_first_seen_uses_each_sources_actual_retrieval_time(self):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [self._snapshot("source_a", "house_a", baseline, [])],
            baseline + pd.Timedelta(minutes=5),
        )
        retrieved = pd.Timestamp("2025-01-03T01:30:00Z")
        snapshot = self._snapshot(
            "source_a",
            "house_a",
            retrieved,
            [self._post("signal", retrieved - pd.Timedelta(minutes=10), "Bullish $NVDA")],
        )

        ingest_snapshots(
            self.connection,
            [snapshot],
            retrieved + pd.Timedelta(minutes=20),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(event["first_seen_utc"], retrieved.isoformat())

    def test_horizons_mature_incrementally_and_are_immutable(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        sessions = pd.date_range("2025-01-03", periods=61, freq="B")
        prices = {
            "NVDA": pd.Series(100.0 + np.arange(61), index=sessions),
            "SOXX": pd.Series(100.0, index=sessions),
            "QQQ": pd.Series(100.0, index=sessions),
        }
        as_of_20 = self._day_after_at_18z(sessions[20])

        evaluate_open_events(self.connection, prices, as_of_20)

        horizons = {
            row["horizon"]: row
            for row in self.connection.execute("SELECT * FROM horizons").fetchall()
        }
        self.assertEqual(horizons[5]["status"], "MATURED")
        self.assertEqual(horizons[20]["status"], "MATURED")
        self.assertEqual(horizons[60]["status"], "PENDING")
        self.assertAlmostEqual(horizons[5]["net_soxx_alpha"], 0.048)
        original_five_day_price = horizons[5]["stock_exit_price"]

        revised = dict(prices)
        revised["NVDA"] = prices["NVDA"] * 0.5
        revised["NVDA"].loc[sessions[5]] = 999.0
        evaluate_open_events(
            self.connection,
            revised,
            self._day_after_at_18z(sessions[60]),
        )

        horizons = {
            row["horizon"]: row
            for row in self.connection.execute("SELECT * FROM horizons").fetchall()
        }
        self.assertEqual(horizons[5]["stock_exit_price"], original_five_day_price)
        self.assertEqual(horizons[60]["status"], "MATURED")
        self.assertEqual(horizons[60]["stock_entry_price"], 50.0)
        self.assertEqual(horizons[60]["stock_exit_price"], 80.0)
        self.assertAlmostEqual(horizons[60]["net_soxx_alpha"], 0.598)
        event = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(event["status"], "COMPLETE")

    def test_missing_prices_remain_in_ledger_and_gate_denominator(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        dates = pd.date_range("2025-01-03", periods=30, freq="B")

        evaluate_open_events(
            self.connection,
            {
                "SOXX": pd.Series(100.0, index=dates),
                "QQQ": pd.Series(100.0, index=dates),
            },
            pd.Timestamp("2025-03-01T18:00:00Z"),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        errors = self.connection.execute(
            "SELECT last_error FROM horizons ORDER BY horizon"
        ).fetchall()
        gate = build_forward_gate(self.connection)
        self.assertEqual(event["status"], "DATA_ERROR")
        self.assertTrue(all("NVDA" in row["last_error"] for row in errors))
        self.assertEqual(gate["metrics"]["admitted_events_total"], 1)
        self.assertEqual(gate["metrics"]["matured_20d_events"], 0)
        self.assertEqual(gate["metrics"]["data_error_events"], 1)
        self.assertFalse(gate["production_signal_enabled"])

    def test_missing_benchmark_entry_on_later_fetch_becomes_data_error(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        dates = pd.date_range("2025-01-03", periods=30, freq="B")
        initial = {
            "NVDA": pd.Series(100.0, index=dates[:2]),
            "SOXX": pd.Series(100.0, index=dates[:2]),
            "QQQ": pd.Series(100.0, index=dates[:2]),
        }
        evaluate_open_events(
            self.connection,
            initial,
            self._day_after_at_18z(dates[1]),
        )
        later = {
            "NVDA": pd.Series(100.0, index=dates),
            "SOXX": pd.Series(100.0, index=dates[1:]),
            "QQQ": pd.Series(100.0, index=dates),
        }

        evaluate_open_events(
            self.connection,
            later,
            self._day_after_at_18z(dates[-1]),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        self.assertEqual(event["status"], "DATA_ERROR")
        self.assertIn("SOXX", event["data_error"])

    def test_missing_stock_entry_after_benchmark_session_is_data_error(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        benchmark_dates = pd.date_range("2025-01-03", periods=3, freq="B")

        evaluate_open_events(
            self.connection,
            {
                "NVDA": pd.Series([90.0], index=pd.to_datetime(["2025-01-02"])),
                "SOXX": pd.Series(100.0, index=benchmark_dates),
                "QQQ": pd.Series(100.0, index=benchmark_dates),
            },
            self._day_after_at_18z(benchmark_dates[-1]),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        gate = build_forward_gate(self.connection)
        self.assertEqual(event["status"], "DATA_ERROR")
        self.assertIn("missing_stock_entry_after_due_session", event["data_error"])
        self.assertEqual(gate["metrics"]["overdue_price_events"], 1)
        self.assertFalse(gate["checks"]["no_overdue_price_events"])

    def test_overdue_stock_horizon_is_data_error_not_survivor_drop(self):
        first_seen = pd.Timestamp("2025-01-03T01:30:00Z")
        self._admit_signal(first_seen)
        stock_dates = pd.date_range("2025-01-03", periods=10, freq="B")
        benchmark_dates = pd.date_range("2025-01-03", periods=30, freq="B")

        evaluate_open_events(
            self.connection,
            {
                "NVDA": pd.Series(100.0, index=stock_dates),
                "SOXX": pd.Series(100.0, index=benchmark_dates),
                "QQQ": pd.Series(100.0, index=benchmark_dates),
            },
            self._day_after_at_18z(benchmark_dates[-1]),
        )

        event = self.connection.execute("SELECT * FROM events").fetchone()
        horizons = {
            row["horizon"]: row
            for row in self.connection.execute("SELECT * FROM horizons").fetchall()
        }
        gate = build_forward_gate(self.connection)
        self.assertEqual(horizons[5]["status"], "MATURED")
        self.assertEqual(horizons[20]["status"], "PENDING")
        self.assertEqual(horizons[20]["last_error"], "stock_horizon_overdue_20d")
        self.assertEqual(event["status"], "DATA_ERROR")
        self.assertEqual(gate["metrics"]["overdue_price_events"], 1)
        self.assertFalse(gate["production_signal_enabled"])

    def test_failed_handle_cache_is_not_ingested_or_checkpointed(self):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [
                self._snapshot("source_a", "house_a", baseline, []),
                self._snapshot("source_b", "house_b", baseline, []),
            ],
            baseline,
        )
        next_poll = pd.Timestamp("2025-01-02T00:00:00Z")
        snapshots = [
            self._snapshot(
                "source_a",
                "house_a",
                next_poll,
                [self._post("a", next_poll - pd.Timedelta(minutes=5), "Bullish $NVDA")],
            ),
            self._snapshot(
                "source_b",
                "house_b",
                baseline,
                [self._post("stale", baseline, "Bullish $AMD")],
            ),
        ]

        ingest_snapshots(
            self.connection,
            snapshots,
            next_poll,
            failed_handles={"source_b"},
        )

        states = {
            row["handle"]: row
            for row in self.connection.execute("SELECT * FROM source_state").fetchall()
        }
        self.assertEqual(states["source_a"]["last_successful_poll_utc"], next_poll.isoformat())
        self.assertEqual(states["source_b"]["last_successful_poll_utc"], baseline.isoformat())
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM posts WHERE handle = 'source_b'"
            ).fetchone()[0],
            0,
        )

    def test_import_is_isolated_from_execution_modules(self):
        code = """
import sys
import research.source_signal_observer
forbidden = {
    'main',
    'allocator_observer',
    'long_short_executor',
    'exchange',
    'backtesting.market_regime_allocator',
}
loaded = forbidden.intersection(sys.modules)
assert not loaded, sorted(loaded)
"""
        subprocess.run([sys.executable, "-c", code], check=True)
        self.assertFalse(build_forward_gate(self.connection)["production_signal_enabled"])

    def test_multi_ticker_post_cannot_satisfy_independent_sample_gate(self):
        seen = "2025-01-02T00:00:00+00:00"
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO posts
                    (handle, tweet_id, post_created_at_utc, first_seen_utc, text,
                     text_sha256, source_universe_version)
                VALUES ('source_a', 'one_post', ?, ?, 'signal', 'hash', 'v1')
                """,
                (seen, seen),
            )
            for number in range(20):
                event_id = f"event-{number}"
                self.connection.execute(
                    """
                    INSERT INTO events
                        (event_id, handle, house, tweet_id, ticker, direction,
                         direction_reason, post_created_at_utc, first_seen_utc,
                         text_sha256, classifier_version, source_universe_version,
                         status, admitted, source_independent, house_independent)
                    VALUES (?, 'source_a', 'house_a', 'one_post', ?, 1,
                            'positive', ?, ?, 'hash', 'v1', 'v1', 'COMPLETE', 1, 1, 1)
                    """,
                    (event_id, f"T{number}", seen, seen),
                )
                self.connection.execute(
                    """
                    INSERT INTO horizons
                        (event_id, horizon, status, net_soxx_alpha,
                         calculation_version)
                    VALUES (?, 20, 'MATURED', 0.10, 'v1')
                    """,
                    (event_id,),
                )

        gate = build_forward_gate(self.connection)

        self.assertEqual(gate["metrics"]["matured_20d_events"], 20)
        self.assertEqual(gate["metrics"]["matured_20d_signal_posts"], 1)
        self.assertEqual(gate["metrics"]["matured_20d_observation_days"], 1)
        self.assertFalse(gate["checks"]["at_least_20_matured_signal_posts"])
        self.assertEqual(gate["phase"], "COLLECTING_FORWARD_DATA")

    def _admit_signal(self, first_seen):
        baseline = pd.Timestamp("2025-01-01T00:00:00Z")
        ingest_snapshots(
            self.connection,
            [self._snapshot("source_a", "house_a", baseline, [])],
            baseline,
        )
        snapshot = self._snapshot(
            "source_a",
            "house_a",
            first_seen,
            [
                self._post(
                    "signal",
                    first_seen - pd.Timedelta(minutes=30),
                    "Bullish $NVDA",
                )
            ],
        )
        ingest_snapshots(self.connection, [snapshot], first_seen)

    @staticmethod
    def _snapshot(handle, house, retrieved_at, posts):
        return {
            "source": {
                "handle": handle,
                "house": house,
                "role": "idea",
                "directional_candidate": True,
            },
            "retrieved_at_utc": pd.Timestamp(retrieved_at).isoformat(),
            "posts": posts,
        }

    @staticmethod
    def _post(tweet_id, created_at, text):
        return {
            "tweet_id": tweet_id,
            "created_at": pd.Timestamp(created_at).isoformat(),
            "text": text,
            "is_reply": False,
            "is_retweet": False,
        }

    @staticmethod
    def _day_after_at_18z(date):
        return pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(days=1, hours=18)

    def _count(self, table):
        return self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


if __name__ == "__main__":
    unittest.main()
