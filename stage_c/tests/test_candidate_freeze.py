import hashlib
import json
import unittest
from pathlib import Path

from stage_c.freeze_stage_c_candidate import canonical_digest, sha256_file


class CandidateFreezeTest(unittest.TestCase):
    def test_canonical_digest_is_order_independent_for_object_keys(self) -> None:
        left = [{"path": "a", "bytes": 1, "sha256": "x"}]
        right = [{"sha256": "x", "bytes": 1, "path": "a"}]
        self.assertEqual(canonical_digest(left), canonical_digest(right))

    def test_sha256_file_and_verify_entries(self) -> None:
        root = Path(__file__).resolve().parents[2]
        path = root / "stage_c/configs/recommended_v2_freeze_c3.json"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(sha256_file(path), expected)

    def test_freeze_policy_is_locked_and_screening_unread(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "stage_c/configs/recommended_v2_freeze_c3.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["freeze_status"], "LOCKED_BEFORE_SCREENING")
        self.assertEqual(config["screening_data_status"], "NOT_ACQUIRED_NOT_READ_NOT_USED")
        self.assertEqual(config["candidate"]["seeds"], [20260723, 20260724, 20260725])
        self.assertEqual([item["weight"] for item in config["candidate"]["components"]], [0.5, 0.5])
        self.assertTrue(config["screening_execution_policy"]["authorization_required_before_data_access"])


if __name__ == "__main__":
    unittest.main()
