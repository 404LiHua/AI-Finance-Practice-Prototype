from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class F22SingleSeedEngineeringTest(unittest.TestCase):
    def test_scope_is_one_candidate_one_seed_three_folds(self) -> None:
        config = json.loads(
            (ROOT / "stage_f/configs/f2_2_single_seed_engineering_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["candidate_id"], "stock_node_gwnet_bounded_cwgan_gp_l8")
        self.assertEqual(config["seed"], 20260725)
        self.assertEqual(config["folds"], ["E_RO_01", "E_RO_02", "E_RO_03"])
        self.assertFalse(config["restrictions"]["additional_seed_execution_allowed"])
        self.assertFalse(config["restrictions"]["screening_allowed"])
        self.assertFalse(config["restrictions"]["final_allowed"])

    def test_existing_receipt_retains_engineering_only_scope(self) -> None:
        path = ROOT / "outputs/stage_f/f2_2_single_seed_engineering_v1/metadata.json"
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["ranking_performed"])
            self.assertFalse(metadata["additional_seed_executed"])
            self.assertFalse(metadata["screening_accessed"])
            self.assertFalse(metadata["final_accessed"])


if __name__ == "__main__":
    unittest.main()
