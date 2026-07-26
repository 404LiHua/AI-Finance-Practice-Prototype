from __future__ import annotations

import unittest

import pandas as pd

from stage_e.build_expanded_panel import broad_industry, select_stratified, week_end


class ExpandedPanelTests(unittest.TestCase):
    def test_week_end_normalizes_holiday_observation(self) -> None:
        result = week_end(pd.Series(["2023-05-04", "2023-05-05"]))
        self.assertEqual(result.iloc[0], pd.Timestamp("2023-05-05"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2023-05-05"))

    def test_industry_mapping(self) -> None:
        self.assertEqual(broad_industry("软件服务"), "信息技术")
        self.assertEqual(broad_industry("银行"), "金融")
        self.assertEqual(broad_industry("化学制药"), "医疗健康")

    def test_stratified_selection_is_deterministic(self) -> None:
        rows = []
        for index in range(18):
            rows.append({
                "stock_code": f"{index:06d}.SZ", "industry_group": f"g{index % 3}",
                "market_cap_bucket_cutoff": ("small", "mid", "large")[index % 3],
                "coverage_ratio": 1.0,
            })
        frame = pd.DataFrame(rows)
        first = select_stratified(frame, 9)["stock_code"].tolist()
        second = select_stratified(frame.sample(frac=1, random_state=7), 9)["stock_code"].tolist()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
