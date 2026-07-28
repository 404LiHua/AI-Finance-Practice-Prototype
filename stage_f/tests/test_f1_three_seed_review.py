from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class F1ThreeSeedReviewTest(unittest.TestCase):
    def test_config_adds_only_authorized_seeds_and_keeps_frozen_scope(self) -> None:
        config = json.loads(
            (ROOT / "stage_f/configs/f1_3_three_seed_review_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["additional_seeds"], [20260723, 20260724])
        self.assertEqual(config["all_seeds"], [20260723, 20260724, 20260725])
        self.assertFalse(config["restrictions"]["code_change_allowed"])
        self.assertFalse(config["restrictions"]["fold_change_allowed"])
        self.assertFalse(config["restrictions"]["training_parameter_change_allowed"])

    def test_three_seed_engineering_contract_is_complete(self) -> None:
        metadata = json.loads(
            (ROOT / "outputs/stage_f/f1_3_three_seed_review_v1/metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["status"], "PASS")
        self.assertEqual(metadata["additional_run_count"], 18)
        self.assertEqual(metadata["all_three_seed_run_count"], 27)
        self.assertEqual(metadata["prediction_rows"], 13500)
        self.assertTrue(metadata["prediction_contract_pass"])
        self.assertTrue(metadata["all_independent_loads_pass"])
        self.assertFalse(metadata["ranking_performed"])

    def test_stability_failures_are_retained_without_promotion(self) -> None:
        acceptance = json.loads(
            (ROOT / "outputs/stage_f/f1_3_three_seed_acceptance_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(acceptance["status"], "ENGINEERING_PASS_STABILITY_HARD_GATE_FAIL")
        self.assertEqual(acceptance["candidates_passing_all_stability_gates"], 0)
        self.assertTrue(all(
            item["failed_gates"] == ["pairwise_pearson", "pairwise_spearman"]
            for item in acceptance["stability_results_in_frozen_candidate_order"]
        ))
        self.assertFalse(acceptance["ranking_performed"])
        self.assertFalse(acceptance["promotion_recommendation_formed"])


if __name__ == "__main__":
    unittest.main()
