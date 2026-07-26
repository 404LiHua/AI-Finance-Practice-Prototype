from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class E6CandidateGateTest(unittest.TestCase):
    def test_gate_is_frozen_before_ranking(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e6_candidate_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "CANDIDATE_GATES_FROZEN_BEFORE_ANY_ELIGIBILITY_OR_RANKING")
        self.assertEqual(len(config["candidate_model_ids"]), 8)
        self.assertFalse(config["eligibility_logic"]["threshold_relaxation_allowed"])
        self.assertFalse(config["restrictions"]["candidate_ranking_before_freeze_receipt"])

    def test_all_requested_gate_families_are_present(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e6_candidate_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["gates"]), {
            "overall", "worst_fold", "per_stock", "industry", "market_cap",
            "return_tails", "seed_stability", "engineering_cost",
        })
        self.assertEqual(config["gates"]["per_stock"]["minimum_stocks_with_mae_below_naive"], 70)
        self.assertEqual(config["gates"]["seed_stability"]["minimum_all_pairwise_prediction_pearson"], 0.80)
        self.assertEqual(config["gates"]["engineering_cost"]["required_inference_receipt_count"], 9)

    def test_three_seed_aggregation_and_tie_break_are_fixed(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e6_candidate_gate_v1.json").read_text(encoding="utf-8"))
        aggregation = config["three_seed_inference_aggregation"]
        self.assertEqual(aggregation["method"], "arithmetic_mean")
        self.assertEqual(aggregation["seed_order"], [20260723, 20260724, 20260725])
        self.assertFalse(aggregation["post_hoc_calibration_allowed"])
        self.assertEqual(config["eligibility_logic"]["tie_break_priority"][0], "lowest_worst_fold_mae")


if __name__ == "__main__":
    unittest.main()
