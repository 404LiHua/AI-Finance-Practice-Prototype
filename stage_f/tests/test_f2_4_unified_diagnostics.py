from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


class F24UnifiedDiagnosticsTest(unittest.TestCase):
    def test_protocol_forbids_retraining_ranking_and_future_data(self) -> None:
        config = json.loads((ROOT / "stage_f/configs/f2_4_unified_robustness_diagnostics_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(config["restrictions"]["new_training_allowed"])
        self.assertFalse(config["restrictions"]["new_model_inference_allowed"])
        self.assertFalse(config["restrictions"]["ranking_allowed"])
        self.assertFalse(config["restrictions"]["stability_failure_compensation_allowed"])
        self.assertFalse(config["restrictions"]["screening_allowed"])
        self.assertFalse(config["restrictions"]["final_allowed"])

    def test_existing_result_retains_gan_stability_hard_failure(self) -> None:
        root = ROOT / "outputs/stage_f/f2_4_unified_robustness_diagnostics_v1"
        if not (root / "metadata.json").is_file():
            return
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        matrix = pd.read_csv(root / "candidate_hard_gate_matrix.csv")
        gan = matrix.loc[matrix["candidate_id"] == "stock_node_gwnet_bounded_cwgan_gp_l8"]
        failures = set(gan.loc[~gan["passed"].astype(bool), "gate_id"].astype(str))
        self.assertTrue(metadata["gan_stability_hard_failure_retained"])
        self.assertIn("stability_pairwise_pearson", failures)
        self.assertIn("stability_pairwise_spearman", failures)
        self.assertIn("stability_prediction_std_mean", failures)
        self.assertIn("stability_prediction_std_p95", failures)
        self.assertFalse(metadata["stability_failure_compensation_allowed"])
        self.assertFalse(metadata["new_training_performed"])
        self.assertFalse(metadata["new_model_inference_performed"])


if __name__ == "__main__":
    unittest.main()
