import json
import unittest

import pandas as pd

from research.source_signal_backtest import (
    build_events,
    build_gate,
    classify_direction,
    classify_ticker_direction,
    extract_tickers,
    parse_embedded_profile,
    run_event_study,
)


class SourceSignalBacktestTest(unittest.TestCase):
    def test_parses_embedded_profile_next_data(self):
        payload = {
            "props": {
                "pageProps": {
                    "timeline": {
                        "entries": [
                            {
                                "content": {
                                    "tweet": {
                                        "id_str": "123",
                                        "created_at": "Wed Jan 01 12:00:00 +0000 2025",
                                        "full_text": "Bullish on $NVDA",
                                        "conversation_id_str": "123",
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        page = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload)
            + "</script>"
        )

        records = parse_embedded_profile(page)

        self.assertEqual(records[0]["tweet_id"], "123")
        self.assertEqual(records[0]["text"], "Bullish on $NVDA")

    def test_extracts_cashtags_and_company_names(self):
        tickers = extract_tickers("NVIDIA benefits more than Intel; $AMD also wins")

        self.assertEqual(tickers, ["AMD", "INTC", "NVDA"])

    def test_requires_unambiguous_direction_language(self):
        self.assertEqual(classify_direction("bullish upside for NVIDIA")[0], 1)
        self.assertEqual(classify_direction("bearish downside for Intel")[0], -1)
        self.assertEqual(classify_direction("bullish but meaningful downside")[0], 0)
        self.assertEqual(classify_direction("NVIDIA launched a chip")[0], 0)

    def test_ticker_direction_ignores_narrative_transactions(self):
        self.assertEqual(
            classify_ticker_direction("The startup sold its business to Microsoft", "MSFT")[0],
            0,
        )
        self.assertEqual(
            classify_ticker_direction("They will be buying TPUs from Broadcom", "AVGO")[0],
            0,
        )
        self.assertEqual(classify_ticker_direction("I would buy $NVDA", "NVDA")[0], 1)

    def test_ticker_direction_separates_multi_ticker_post(self):
        text = "$WULF - ER was positive. $CRWV - ER was bearish."

        self.assertEqual(classify_ticker_direction(text, "WULF")[0], 1)
        self.assertEqual(classify_ticker_direction(text, "CRWV")[0], -1)

    def test_cooldown_deduplicates_repeated_source_signal(self):
        posts = pd.DataFrame(
            [
                self._post("2025-01-01", "1"),
                self._post("2025-01-10", "2"),
                self._post("2025-02-05", "3"),
            ]
        )

        events = build_events(posts)

        self.assertEqual(events["source_independent"].tolist(), [True, False, True])

    def test_event_study_enters_strictly_after_post_date_and_applies_cost(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="B")
        prices = {
            "NVDA": pd.Series(range(100, 130), index=dates, dtype=float),
            "SOXX": pd.Series(100.0, index=dates),
            "QQQ": pd.Series(100.0, index=dates),
        }
        event = pd.DataFrame(
            [
                {
                    "handle": "source",
                    "house": "house",
                    "tweet_id": "1",
                    "created_at_utc": pd.Timestamp("2025-01-01 12:00", tz="UTC"),
                    "ticker": "NVDA",
                    "direction": 1,
                    "direction_reason": "bullish",
                    "is_reply": False,
                    "is_retweet": False,
                    "source_independent": True,
                    "house_independent": True,
                }
            ]
        )

        result = run_event_study(event, prices, horizons=(5,), cost_bps=20)

        self.assertEqual(result.iloc[0]["entry_date"], pd.Timestamp("2025-01-02"))
        self.assertEqual(result.iloc[0]["exit_date_5d"], pd.Timestamp("2025-01-09"))
        self.assertEqual(result.iloc[0]["exit_price_5d"], 106.0)
        expected = 106 / 101 - 1 - 0.002
        self.assertAlmostEqual(result.iloc[0]["net_soxx_alpha_5d"], expected)

    def test_gate_rejects_small_sample_even_if_return_is_positive(self):
        summary = pd.DataFrame(
            [
                {
                    "handle": "ALL_INDEPENDENT",
                    "events": 5,
                    "tickers": 2,
                    "span_days": 100,
                    "median_net_soxx_alpha": 0.10,
                    "win_rate": 0.80,
                    "bootstrap_mean_ci_low": 0.02,
                    "max_ticker_share": 0.40,
                }
            ]
        )

        gate = build_gate(summary)

        self.assertEqual(gate["status"], "INSUFFICIENT_DATA")
        self.assertFalse(gate["production_signal_enabled"])

    @staticmethod
    def _post(created_at, tweet_id):
        return {
            "handle": "source",
            "house": "house",
            "role": "idea",
            "directional_candidate": True,
            "tweet_id": tweet_id,
            "created_at_utc": pd.Timestamp(created_at, tz="UTC"),
            "text": "Bullish $NVDA",
            "is_reply": False,
            "is_retweet": False,
            "direction": 1,
            "direction_reason": "bullish",
            "tickers": ["NVDA"],
        }


if __name__ == "__main__":
    unittest.main()
