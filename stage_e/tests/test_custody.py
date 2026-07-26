import unittest
from pathlib import Path

import pandas as pd

from stage_e.custody import StageEDataCustodyGuard, StageEDataCustodyViolation


class StageECustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.guard = StageEDataCustodyGuard.from_config(
            cls.root / "stage_e/configs/data_custody_v1.json", cls.root
        )

    def test_c4_and_d5_paths_are_rejected(self):
        for path in (
            self.root / "data/screening/c4.csv",
            self.root / "outputs/stage_d/d5_screening_20240614_20250613/result.csv",
        ):
            with self.assertRaises(StageEDataCustodyViolation):
                self.guard.assert_path_allowed(path)

    def test_sealed_and_post_ceiling_dates_are_rejected(self):
        for date in ("2023-06-09", "2024-06-14", "2025-06-20"):
            frame = pd.DataFrame({"trade_date": [date], "target_date": [date]})
            with self.assertRaises(StageEDataCustodyViolation):
                self.guard.assert_development_frame(frame)

    def test_development_dates_are_allowed(self):
        self.guard.assert_development_frame(pd.DataFrame({
            "trade_date": ["2023-05-19"], "target_date": ["2023-05-26"]
        }))


if __name__ == "__main__":
    unittest.main()
