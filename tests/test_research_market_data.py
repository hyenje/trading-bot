import unittest
from unittest.mock import Mock, patch

import pandas as pd

from research.market_data import fetch_yahoo_adjusted_close


class ResearchMarketDataTest(unittest.TestCase):
    @patch("research.market_data.requests.get")
    def test_raw_close_is_not_silently_used_when_adjusted_close_is_missing(
        self, get
    ):
        response = Mock()
        response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1_735_689_600],
                        "indicators": {"quote": [{"close": [100.0]}]},
                    }
                ]
            }
        }
        get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "adjusted-close data missing"):
            fetch_yahoo_adjusted_close(
                "NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")
            )


if __name__ == "__main__":
    unittest.main()
