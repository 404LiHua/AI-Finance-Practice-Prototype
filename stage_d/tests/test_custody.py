import unittest
from pathlib import Path

import pandas as pd

from stage_d.custody import DataCustodyGuard, DataCustodyViolation


class DataCustodyGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.guard = DataCustodyGuard.from_config(
            cls.root / "stage_d/configs/data_custody.json", cls.root
        )

    def test_c4_path_is_rejected_before_read(self) -> None:
        sealed = self.root / "data/screening/anything.csv"
        with self.assertRaises(DataCustodyViolation):
            self.guard.assert_path_allowed(sealed)

    def test_c4_identifier_is_rejected_outside_default_root(self) -> None:
        sealed = self.root / "tmp/stage_c_recommended_v2_c4_20230609_20240607.csv"
        with self.assertRaises(DataCustodyViolation):
            self.guard.assert_path_allowed(sealed)

    def test_development_frame_is_allowed(self) -> None:
        frame = pd.DataFrame({
            "trade_date": ["2023-05-19"],
            "target_date": ["2023-05-26"],
        })
        self.guard.assert_development_frame(frame)

    def test_date_beyond_ceiling_is_rejected(self) -> None:
        frame = pd.DataFrame({
            "trade_date": ["2023-06-09"],
            "target_date": ["2023-06-16"],
        })
        with self.assertRaises(DataCustodyViolation):
            self.guard.assert_development_frame(frame)


if __name__ == "__main__":
    unittest.main()
