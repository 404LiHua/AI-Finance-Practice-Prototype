from __future__ import annotations

import json
import unittest
from pathlib import Path

from stage_d.freeze_stage_d_candidate import FREEZE_DIR, verify_freeze


REPO_ROOT = Path(__file__).resolve().parents[2]


class D4FreezeTest(unittest.TestCase):
    def test_freeze_manifest_verifies(self) -> None:
        result = verify_freeze()
        self.assertTrue(result["verified"])
        self.assertEqual(result["artifact_count"], 31)

    def test_inference_manifest_is_exact(self) -> None:
        manifest = json.loads((FREEZE_DIR / "INFERENCE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["candidate_model"], "frets_return_l4__fixed_shrink_a075")
        self.assertEqual(manifest["shrinkage_alpha"], 0.75)
        self.assertEqual(manifest["aggregation"], "arithmetic_mean")
        self.assertEqual([item["seed"] for item in manifest["checkpoints"]], [
            20260723, 20260724, 20260725,
        ])

    def test_independent_recalc_receipt(self) -> None:
        path = REPO_ROOT / "outputs/stage_d/d4_independent_recalc_v1/INDEPENDENT_RECALC_RECEIPT.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["independent_load"], "PASS")
        self.assertEqual(receipt["independent_prediction_recalc"], "PASS")
        self.assertEqual(receipt["c4_rows_read"], 0)
        self.assertEqual(receipt["future_d_screening_rows_read"], 0)


if __name__ == "__main__":
    unittest.main()
