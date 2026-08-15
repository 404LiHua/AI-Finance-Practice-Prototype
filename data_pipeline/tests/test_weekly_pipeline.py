import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_weekly_dataset.py"
SPEC = importlib.util.spec_from_file_location("weekly_pipeline", MODULE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


class WeeklyPipelineTests(unittest.TestCase):
    def sample(self):
        rows = []
        for code in ("000001.SZ", "000002.SZ"):
            for index, date in enumerate(pd.date_range("2020-01-03", periods=40, freq="W-FRI")):
                close = 10 + index * 0.1
                rows.append({
                    "stock_code": code,
                    "trade_date": date,
                    "close": close,
                    "open": close - 0.05,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "previous_close": close - 0.1,
                    "price_change": 0.1,
                    "return_reported": 0.01,
                    "volume_hands": 1000 + index,
                    "amount_thousand_cny": 10000 + index,
                    "source_file": f"{code}.csv",
                    "source_sha256": "x",
                    "source_row_number": index + 2,
                })
        return pd.DataFrame(rows)

    def test_calendar_and_features_are_past_only(self):
        result = PIPELINE.add_calendar_and_features(self.sample())
        first = result[result.stock_code == "000001.SZ"].iloc[0]
        self.assertTrue(pd.isna(first["return_1w"]))
        self.assertEqual(first["calendar_week_end"], first["trade_date"])

    def test_target_never_crosses_split_boundary(self):
        config = {
            "split": {
                "train_ratio": 0.7,
                "validation_ratio": 0.1,
                "test_ratio": 0.2,
                "purge_weeks": 1,
            },
            "task": {"forecast_horizon_weeks": 1, "lookback_weeks": 3},
        }
        result = PIPELINE.add_calendar_and_features(self.sample())
        result, _ = PIPELINE.assign_splits(result, config)
        eligible = result[result.sample_eligible]
        self.assertGreater(len(eligible), 0)
        self.assertTrue((eligible.history_weeks_available >= 3).all())
        lookup = result.set_index(["stock_code", "calendar_week_end"])["split"]
        for row in eligible.itertuples():
            self.assertEqual(row.split, lookup.loc[(row.stock_code, row.target_date)])

    def test_stock_basic_join_preserves_rows_and_listing_age(self):
        prices = self.sample()
        basic = pd.DataFrame({
            "stock_code": ["000001.SZ", "000002.SZ"],
            "stock_name": ["A", "B"],
            "listing_date": pd.to_datetime(["2010-01-01", "2011-01-01"]),
        })
        merged = PIPELINE.merge_stock_basic(prices, basic)
        self.assertEqual(len(merged), len(prices))
        self.assertTrue(merged.stock_basic_available.all())
        self.assertFalse(merged.before_listing_date.any())
        self.assertTrue((merged.weeks_since_listing >= 0).all())


if __name__ == "__main__":
    unittest.main()
