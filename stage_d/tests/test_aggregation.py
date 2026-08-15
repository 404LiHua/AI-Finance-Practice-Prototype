import unittest

import pandas as pd

from stage_d.aggregation import aggregate_cross_fold, validate_metric_grid


class CrossFoldAggregationTest(unittest.TestCase):
    def metrics(self) -> pd.DataFrame:
        rows = []
        for fold_index, fold in enumerate(("D_RO_01", "D_RO_02", "D_RO_03")):
            for seed in (1, 2):
                naive_mae = 0.04 + fold_index * 0.002
                rows.append({
                    "model": "naive", "fold_id": fold, "seed": seed, "samples": 100,
                    "mae": naive_mae, "rmse": naive_mae * 1.4,
                    "direction_accuracy": 0.5, "direction_f1": 0.0,
                })
                rows.append({
                    "model": "candidate", "fold_id": fold, "seed": seed, "samples": 100,
                    "mae": naive_mae * (0.95 if fold_index < 2 else 1.01),
                    "rmse": naive_mae * 1.35,
                    "direction_accuracy": 0.55, "direction_f1": 0.4,
                })
        return pd.DataFrame(rows)

    def test_summary_counts_fold_wins_and_worst_gap(self) -> None:
        per_fold, summary, metadata = aggregate_cross_fold(self.metrics())
        candidate = summary.set_index("model").loc["candidate"]
        self.assertEqual(candidate["folds_beating_baseline_mae"], 2)
        self.assertAlmostEqual(candidate["fold_win_rate"], 2 / 3)
        self.assertAlmostEqual(candidate["worst_fold_mae_gap_vs_baseline_pct"], 1.0)
        self.assertEqual(metadata["folds"], ["D_RO_01", "D_RO_02", "D_RO_03"])
        self.assertEqual(len(per_fold), 6)

    def test_incomplete_grid_is_rejected(self) -> None:
        frame = self.metrics().iloc[:-1].copy()
        with self.assertRaises(ValueError):
            validate_metric_grid(frame)


if __name__ == "__main__":
    unittest.main()
