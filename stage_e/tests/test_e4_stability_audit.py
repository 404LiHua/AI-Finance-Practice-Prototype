from __future__ import annotations

import unittest

import pandas as pd

from stage_e.run_e4_stability_audit import audit_prediction_variant, frame_key_sha256, pair_metrics


class E4StabilityAuditTest(unittest.TestCase):
    def _frame(self, seed: int, shift: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame({
            "variant": ["v"] * 4, "fold_id": ["f"] * 4,
            "trade_date": ["2023-01-01", "2023-01-01", "2023-01-08", "2023-01-08"],
            "stock_code": ["a", "b", "a", "b"],
            "target_return": [0.1, -0.1, 0.2, -0.2],
            "prediction": [0.08 + shift, -0.08 + shift, 0.16 + shift, -0.16 + shift],
            "sample_valid": [True] * 4, "text_available": [False] * 4, "seed": [seed] * 4,
        })

    def test_pair_metrics_preserve_keys(self) -> None:
        left, right = self._frame(1), self._frame(2, 0.01)
        self.assertEqual(frame_key_sha256(left), frame_key_sha256(right))
        metrics = pair_metrics(left, right)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["pearson"], 1.0)

    def test_variant_audit_emits_all_sections(self) -> None:
        frame = pd.concat([self._frame(1), self._frame(2, 0.01), self._frame(3, -0.01)], ignore_index=True)
        audit = audit_prediction_variant("v", frame, [1, 2, 3], ["f"])
        self.assertEqual(len(audit["key_rows"]), 3)
        self.assertEqual(len(audit["pairwise_overall"]), 3)
        self.assertEqual(len(audit["pairwise_by_fold"]), 3)


if __name__ == "__main__":
    unittest.main()
