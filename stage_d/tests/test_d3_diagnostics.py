from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from stage_d.d3_diagnostics import (
    assign_return_group,
    load_diagnostic_config,
    select_unique_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "stage_d/configs/d3_diagnostics.json"


class D3DiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_diagnostic_config(CONFIG_PATH, REPO_ROOT)

    def test_fixed_return_groups_cover_boundaries(self) -> None:
        values = pd.Series([-0.04, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
        observed = assign_return_group(values, self.config["return_groups"]).tolist()
        self.assertEqual(observed, [
            "negative_tail", "negative_moderate", "near_zero", "near_zero",
            "near_zero", "positive_moderate", "positive_moderate", "positive_tail",
        ])

    def test_robust_gate_and_order_are_deterministic(self) -> None:
        frame = pd.DataFrame([
            {"model": "naive", "folds_beating_baseline_mae": 0,
             "worst_fold_mae_gap_vs_baseline_pct": 0.0,
             "mean_mae_improvement_vs_baseline_pct": 0.0, "mae_mean": 1.0,
             "mae_cv": 0.0, "rmse_mean": 1.0},
            {"model": "candidate_b", "folds_beating_baseline_mae": 3,
             "worst_fold_mae_gap_vs_baseline_pct": -1.0,
             "mean_mae_improvement_vs_baseline_pct": 2.0, "mae_mean": 0.90,
             "mae_cv": 0.2, "rmse_mean": 0.95},
            {"model": "candidate_a", "folds_beating_baseline_mae": 2,
             "worst_fold_mae_gap_vs_baseline_pct": 1.0,
             "mean_mae_improvement_vs_baseline_pct": 1.0, "mae_mean": 0.89,
             "mae_cv": 0.1, "rmse_mean": 0.94},
            {"model": "rejected", "folds_beating_baseline_mae": 1,
             "worst_fold_mae_gap_vs_baseline_pct": 1.0,
             "mean_mae_improvement_vs_baseline_pct": 3.0, "mae_mean": 0.80,
             "mae_cv": 0.1, "rmse_mean": 0.90},
        ])
        _, selected = select_unique_candidate(frame, self.config)
        self.assertEqual(selected["recommendation"]["recommended_model"], "candidate_a")
        self.assertEqual(len(selected["eligible"]), 2)

    def test_locked_rules_forbid_candidate_or_shrinkage_changes(self) -> None:
        self.assertFalse(self.config["candidate_additions_allowed"])
        self.assertFalse(self.config["shrinkage_changes_allowed"])


if __name__ == "__main__":
    unittest.main()
