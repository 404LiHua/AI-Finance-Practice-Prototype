from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from stage_d.run_d5_screening import END_DATE, START_DATE, _stock_to_baostock


REPO_ROOT = Path(__file__).resolve().parents[2]


class D5ScreeningGuardTest(unittest.TestCase):
    def test_authorized_interval_is_exact(self) -> None:
        authorization = json.loads((
            REPO_ROOT / "stage_d/authorizations/D5_AUTHORIZATION_20260724.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(authorization["scope"]["screening_interval_start"], "2024-06-14")
        self.assertEqual(authorization["scope"]["screening_interval_end"], "2025-06-13")
        self.assertEqual(START_DATE, pd.Timestamp("2024-06-14"))
        self.assertEqual(END_DATE, pd.Timestamp("2025-06-13"))

    def test_stock_code_conversion(self) -> None:
        self.assertEqual(_stock_to_baostock("000001.SZ"), "sz.000001")

    def test_authorization_is_single_use(self) -> None:
        authorization = json.loads((
            REPO_ROOT / "stage_d/authorizations/D5_AUTHORIZATION_20260724.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(authorization["scope"]["execution_count"], 1)
        self.assertEqual(authorization["authorization_text"], "授权执行 D-5")


if __name__ == "__main__":
    unittest.main()
