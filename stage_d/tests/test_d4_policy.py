from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stage_d.d4_policy import evaluate_frozen_policy


REPO_ROOT = Path(__file__).resolve().parents[2]


class D4PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freeze = json.loads((REPO_ROOT / "stage_d/configs/d4_freeze.json").read_text(encoding="utf-8"))
        cls.groups = json.loads((REPO_ROOT / "stage_d/configs/d3_diagnostics.json").read_text(encoding="utf-8"))[
            "return_groups"
        ]

    def test_frozen_identity_and_aggregation(self) -> None:
        candidate = self.freeze["candidate"]
        self.assertEqual(candidate["model_id"], "frets_return_l4__fixed_shrink_a075")
        self.assertEqual(candidate["shrinkage_alpha"], 0.75)
        self.assertEqual(candidate["seeds"], [20260723, 20260724, 20260725])
        self.assertEqual(candidate["seed_aggregation"], "arithmetic_mean_after_per_seed_shrinkage")

    def test_policy_passes_clear_synthetic_improvement(self) -> None:
        returns = np.tile(np.asarray([-0.04, -0.02, -0.005, 0.02, 0.04]), 24)
        samples = pd.DataFrame({
            "stock_code": np.repeat([f"S{i:02d}" for i in range(30)], 4),
            "trade_date": pd.date_range("2020-01-01", periods=120),
            "target_date": pd.date_range("2020-01-08", periods=120),
            "model_close": 1.0,
            "target_close": 1.0 + returns,
            "target_return": returns,
            "target_direction": (returns > 0).astype(int),
        })
        candidate = 0.5 * returns
        seeds = {20260723: 0.49 * returns, 20260724: 0.5 * returns, 20260725: 0.51 * returns}
        result = evaluate_frozen_policy(samples, candidate, seeds, self.freeze, self.groups)
        self.assertEqual(result["outcome"], "PASS")

    def test_policy_rejects_wrong_stock_scope(self) -> None:
        samples = pd.DataFrame({
            "stock_code": ["S00"] * 4,
            "target_return": [0.01] * 4,
        })
        result = evaluate_frozen_policy(
            samples, np.zeros(4), {20260723: np.zeros(4), 20260724: np.zeros(4), 20260725: np.zeros(4)},
            self.freeze, self.groups,
        )
        self.assertEqual(result["outcome"], "INVALID_INTEGRITY_FAILURE")


if __name__ == "__main__":
    unittest.main()
