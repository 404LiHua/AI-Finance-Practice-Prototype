from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from stage_f.robustness import F1_CANDIDATE_IDS
from stage_f.run_f1_single_seed import _assert_frozen_keys


ROOT = Path(__file__).resolve().parents[2]


class F1SingleSeedEngineeringTest(unittest.TestCase):
    def test_effective_config_keeps_frozen_scope(self) -> None:
        config = json.loads(
            (ROOT / "stage_f/configs/f1_2_single_seed_engineering_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["seed"], 20260725)
        self.assertEqual(config["folds"], ["E_RO_01", "E_RO_02", "E_RO_03"])
        self.assertEqual(tuple(config["candidate_ids"]), F1_CANDIDATE_IDS)
        self.assertFalse(config["restrictions"]["additional_seed_execution_allowed"])
        self.assertFalse(config["restrictions"]["candidate_ranking_allowed"])
        self.assertFalse(config["restrictions"]["gan_training_allowed"])

    def test_frozen_key_comparison_is_order_invariant_and_exact(self) -> None:
        rows = []
        for index in range(500):
            rows.append({
                "fold_id": "E_RO_01",
                "sample_row_id": f"row-{index}",
                "trade_date": "2022-01-07",
                "target_date": "2022-01-14",
                "stock_code": f"{index % 100:06d}.SZ",
                "sample_valid": True,
            })
        actual = pd.DataFrame(rows)
        expected = actual.sample(frac=1.0, random_state=7).reset_index(drop=True)
        receipt = _assert_frozen_keys(actual, expected, "E_RO_01")
        self.assertEqual(len(receipt), 64)
        changed = expected.copy()
        changed.loc[0, "sample_row_id"] = "changed"
        with self.assertRaises(RuntimeError):
            _assert_frozen_keys(actual, changed, "E_RO_01")

    def test_final_receipt_passes_without_ranking_or_extra_seed(self) -> None:
        acceptance = json.loads(
            (ROOT / "outputs/stage_f/f1_2_single_seed_acceptance_v1.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (ROOT / "outputs/stage_f/f1_2_single_seed_engineering_v2/metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(acceptance["status"], "PASS")
        self.assertEqual(acceptance["passed_checks"], 12)
        self.assertEqual(metadata["completed_run_count"], 9)
        self.assertEqual(metadata["seed"], 20260725)
        self.assertFalse(metadata["ranking_performed"])
        self.assertFalse(metadata["additional_seed_executed"])
        self.assertTrue(metadata["pooled_three_fold_stress_nonempty"])


if __name__ == "__main__":
    unittest.main()
