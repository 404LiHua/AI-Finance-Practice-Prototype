from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from stage_f.custody import StageFDataCustodyGuard, StageFDataCustodyViolation
from stage_f.protocol import fold_key_receipt, validate_stress_contract


ROOT = Path(__file__).resolve().parents[2]


class StageF0ProtocolTest(unittest.TestCase):
    def test_custody_rejects_screening_final_and_post_ceiling(self) -> None:
        guard = StageFDataCustodyGuard.from_config(ROOT / "stage_f/configs/f0_data_custody_v1.json", ROOT)
        for path in (ROOT / "data/screening/f.csv", ROOT / "outputs/stage_f/final/result.json"):
            with self.assertRaises(StageFDataCustodyViolation):
                guard.assert_path_allowed(path)
        with self.assertRaises(StageFDataCustodyViolation):
            guard.assert_development_dates(["2023-06-09"], "trade_date")
        guard.assert_development_dates(["2023-05-26", "2023-06-02"], "trade_date")

    def test_fold_key_hash_is_deterministic(self) -> None:
        arrays = {
            "sample_row_id": np.array([["a", "b"], ["c", "d"]]),
            "split": np.array(["train", "validation"]),
            "trade_date": np.array(["2023-01-06", "2023-01-13"]),
            "target_date": np.array([["2023-01-13", "2023-01-13"], ["2023-01-20", "2023-01-20"]]),
            "stock_code": np.array(["000001.SZ", "000002.SZ"]),
            "sample_mask": np.ones((2, 2), dtype=bool),
        }
        left = fold_key_receipt(arrays, "F_RO_01")
        right = fold_key_receipt({key: value.copy() for key, value in arrays.items()}, "F_RO_01")
        self.assertEqual(left["sample_key_sha256"], right["sample_key_sha256"])
        self.assertEqual(left["validation_valid_sample_count"], 2)

    def test_stress_contract_is_train_only_and_bounded(self) -> None:
        config = json.loads((ROOT / "stage_f/configs/f0_stress_scenarios_v1.json").read_text(encoding="utf-8"))
        validate_stress_contract(config)
        self.assertEqual(len(config["scenarios"]), 9)


if __name__ == "__main__":
    unittest.main()
