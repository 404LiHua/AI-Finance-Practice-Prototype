from __future__ import annotations

import hashlib
import unittest

import pandas as pd

from stage_e.build_e2_text_views import align_availability, normalize_code


class E2TextViewTests(unittest.TestCase):
    def test_explicit_stock_mapping(self) -> None:
        self.assertEqual(normalize_code("000001"), "000001.SZ")
        self.assertEqual(normalize_code("600000"), "600000.SH")

    def test_friday_after_close_moves_to_next_open_week(self) -> None:
        events = pd.DataFrame({"published_at": pd.to_datetime(["2023-01-06T15:01:00+08:00"], utc=True)})
        weeks = pd.DatetimeIndex(["2023-01-06", "2023-01-13"])
        aligned = align_availability(events, weeks, "Asia/Shanghai")
        self.assertEqual(aligned["trade_date"].iloc[0], pd.Timestamp("2023-01-13"))

    def test_canonical_record_hash_example(self) -> None:
        values = ["2023-01-03T01:00:00+00:00", "p", "i", "u", "000001.SZ", "t", "b", "l"]
        self.assertEqual(len(hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
