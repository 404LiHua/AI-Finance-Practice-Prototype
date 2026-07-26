from __future__ import annotations

import unittest

import pandas as pd

from stage_e.run_e4_fixed_graph_stabilization import select_variant


class FixedGraphStabilizationTest(unittest.TestCase):
    def test_selection_uses_frozen_metric_order(self) -> None:
        summary = pd.DataFrame([
            {"variant": "b", "mean_mae": 0.03, "worst_fold_mae": 0.04, "fold_mae_std": 0.002, "adjacency_valid": True, "row_stochastic": True},
            {"variant": "a", "mean_mae": 0.03, "worst_fold_mae": 0.04, "fold_mae_std": 0.001, "adjacency_valid": True, "row_stochastic": True},
            {"variant": "invalid", "mean_mae": 0.01, "worst_fold_mae": 0.02, "fold_mae_std": 0.0, "adjacency_valid": False, "row_stochastic": True},
        ])
        self.assertEqual(select_variant(summary), "a")


if __name__ == "__main__":
    unittest.main()
