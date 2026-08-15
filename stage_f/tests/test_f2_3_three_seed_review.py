from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class F23ThreeSeedReviewTest(unittest.TestCase):
    def test_only_two_remaining_seeds_are_added(self) -> None:
        path = ROOT / "stage_f/configs/f2_3_three_seed_review_v1.json"
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["additional_seeds"], [20260723, 20260724])
            self.assertEqual(config["all_seeds"], [20260723, 20260724, 20260725])
            self.assertFalse(config["restrictions"]["candidate_ranking_allowed"])
            self.assertFalse(config["restrictions"]["screening_allowed"])
            self.assertFalse(config["restrictions"]["final_allowed"])

    def test_existing_metadata_retains_all_seeds_and_failures(self) -> None:
        path = ROOT / "outputs/stage_f/f2_3_three_seed_review_v1/metadata.json"
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["all_seeds"], [20260723, 20260724, 20260725])
            self.assertEqual(metadata["all_three_seed_run_count"] + metadata["failure_count"], 9)
            self.assertFalse(metadata["ranking_performed"])
            self.assertFalse(metadata["screening_accessed"])
            self.assertFalse(metadata["final_accessed"])


if __name__ == "__main__":
    unittest.main()
