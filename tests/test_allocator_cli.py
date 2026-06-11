import unittest
from datetime import date

from main import _allocator_report_days


class AllocatorCliTest(unittest.TestCase):
    def test_allocator_start_date_converts_to_report_days(self):
        days = _allocator_report_days("2019-09-08", date(2026, 6, 9), 1095)

        self.assertEqual(days, 2466)

    def test_allocator_start_date_defaults_to_default_days(self):
        days = _allocator_report_days(None, date(2026, 6, 9), 1095)

        self.assertEqual(days, 1095)

    def test_allocator_start_date_must_be_before_latest_price_date(self):
        with self.assertRaises(ValueError):
            _allocator_report_days("2026-06-09", date(2026, 6, 9), 1095)

    def test_allocator_start_date_requires_iso_date(self):
        with self.assertRaises(ValueError):
            _allocator_report_days("2019/09/08", date(2026, 6, 9), 1095)


if __name__ == "__main__":
    unittest.main()
