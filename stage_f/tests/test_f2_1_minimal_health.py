from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class F21MinimalGanHealthTest(unittest.TestCase):
    def test_authorization_is_minimal_and_future_data_remains_closed(self) -> None:
        config = json.loads(
            (ROOT / "stage_f/configs/f2_1_minimal_training_health_v1.json").read_text(encoding="utf-8")
        )
        self.assertTrue(config["authorization"]["optimizer_steps_authorized"])
        self.assertTrue(config["authorization"]["synthetic_minimal_training_authorized"])
        self.assertFalse(config["authorization"]["formal_fold_training_authorized"])
        self.assertFalse(config["authorization"]["additional_seed_training_authorized"])
        self.assertFalse(config["authorization"]["screening_authorized"])
        self.assertFalse(config["authorization"]["final_authorized"])

    def test_health_receipt_passes_if_present(self) -> None:
        path = ROOT / "outputs/stage_f/f2_1_minimal_training_health_receipt_v1.json"
        if path.is_file():
            receipt = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["passed_checks"], receipt["required_checks"])
            self.assertFalse(receipt["formal_fold_training_executed"])
            self.assertFalse(receipt["screening_accessed"])
            self.assertFalse(receipt["final_accessed"])


if __name__ == "__main__":
    unittest.main()
