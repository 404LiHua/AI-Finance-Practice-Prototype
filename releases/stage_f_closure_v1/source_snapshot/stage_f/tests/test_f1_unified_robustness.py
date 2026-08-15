from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


class F1UnifiedRobustnessTest(unittest.TestCase):
    def test_effective_config_inherits_identical_f0_hard_gates(self) -> None:
        f0 = json.loads((ROOT / "stage_f/configs/f0_robustness_protocol_v1.json").read_text(encoding="utf-8"))
        base = json.loads(
            (ROOT / "stage_f/configs/f1_4_unified_robustness_diagnostics_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(base["hard_gates"], f0["hard_gates"])
        self.assertTrue(base["eligibility"]["stability_failure_is_non_compensable"])

    def test_all_diagnostic_domains_and_gate_rows_exist(self) -> None:
        output = ROOT / "outputs/stage_f/f1_4_unified_robustness_diagnostics_v2"
        for name in (
            "normal_overall.csv", "normal_fold.csv", "normal_per_stock.csv",
            "normal_industry.csv", "normal_market_cap.csv", "normal_return_decile.csv",
            "stress_scenario_metrics.csv", "stability_diagnostics.json", "engineering_costs.csv",
        ):
            self.assertTrue((output / name).is_file(), name)
        matrix = pd.read_csv(output / "candidate_hard_gate_matrix.csv")
        self.assertEqual(len(matrix), 60)
        self.assertEqual(matrix.groupby("candidate_id").size().tolist(), [20, 20, 20])

    def test_formal_no_candidate_conclusion_retains_stability_failures(self) -> None:
        acceptance = json.loads(
            (ROOT / "outputs/stage_f/f1_4_unified_robustness_acceptance_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(acceptance["eligible_candidate_count"], 0)
        self.assertEqual(
            acceptance["eligibility_conclusion"],
            "FORMAL_NO_ROBUST_PROMOTABLE_CANDIDATE_RETAIN_STAGE_E_INCUMBENT",
        )
        self.assertTrue(acceptance["stability_failures_non_compensable"])
        self.assertTrue(all(not item["all_hard_gates_pass"] for item in acceptance["candidate_results_in_frozen_order"]))
        self.assertFalse(acceptance["ranking_performed"])
        self.assertFalse(acceptance["threshold_relaxation_performed"])


if __name__ == "__main__":
    unittest.main()
