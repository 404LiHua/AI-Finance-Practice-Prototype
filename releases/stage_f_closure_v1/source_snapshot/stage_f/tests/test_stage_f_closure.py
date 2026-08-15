from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "releases/stage_f_closure_v1"


class StageFClosureTest(unittest.TestCase):
    def test_frozen_conclusion_retains_all_failures(self) -> None:
        if not (RELEASE / "NEGATIVE_CONCLUSION.json").is_file():
            return
        conclusion = json.loads((RELEASE / "NEGATIVE_CONCLUSION.json").read_text(encoding="utf-8"))
        self.assertEqual(conclusion["promotable_candidate_count"], 0)
        self.assertEqual(conclusion["retained_model"], "stock_node_gwnet_fixed_industry_l8")
        self.assertEqual(len(conclusion["candidate_gate_counts"]), 4)
        self.assertEqual(len(conclusion["gan_non_compensable_stability_failures"]), 4)
        self.assertFalse(conclusion["screening_accessed"])
        self.assertFalse(conclusion["final_accessed"])

    def test_release_manifest_has_source_evidence_and_negative_receipts(self) -> None:
        if not (RELEASE / "SHA256_MANIFEST.json").is_file():
            return
        manifest = json.loads((RELEASE / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
        paths = [item["path"] for item in manifest["entries"]]
        self.assertGreater(manifest["artifact_count"], 150)
        self.assertTrue(any(path.startswith("source_snapshot/stage_f/") for path in paths))
        self.assertTrue(any(path.startswith("evidence_snapshot/outputs/stage_f/") for path in paths))
        self.assertTrue(any("failure" in path.casefold() for path in paths))


if __name__ == "__main__":
    unittest.main()
