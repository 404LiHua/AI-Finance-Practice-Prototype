import unittest

import pandas as pd

from stage_e.hashing import canonical_row_set_sha256, manifest_root_sha256
from stage_e.protocol import FrozenFold, build_frozen_assignments


class StageEProtocolTests(unittest.TestCase):
    def panel(self):
        dates = pd.date_range("2022-01-07", periods=60, freq="W-FRI")
        rows = []
        for code in ("000001.SZ", "000002.SZ"):
            for index, date in enumerate(dates):
                target = dates[index + 1] if index + 1 < len(dates) else pd.NaT
                rows.append({
                    "stock_code": code,
                    "trade_date": date,
                    "target_date": target,
                    "target_return": 0.01 if pd.notna(target) else None,
                    "model_eligible_pit": True,
                    "history_weeks_available": index + 1,
                    "cross_section_eligible": True,
                })
        return pd.DataFrame(rows)

    def test_row_hash_is_order_invariant(self):
        panel = self.panel()
        self.assertEqual(
            canonical_row_set_sha256(panel),
            canonical_row_set_sha256(panel.sample(frac=1.0, random_state=7)),
        )

    def test_manifest_root_is_order_invariant(self):
        records = [
            {"source_id": "b", "relative_path": "2", "sha256": "y", "size_bytes": 2},
            {"source_id": "a", "relative_path": "1", "sha256": "x", "size_bytes": 1},
        ]
        self.assertEqual(manifest_root_sha256(records), manifest_root_sha256(reversed(records)))

    def test_frozen_assignments_do_not_overlap(self):
        folds = [FrozenFold(
            fold_id="E_RO_01",
            train_start_policy="available_history_start",
            train_end="2022-09-30",
            purge_start="2022-10-07",
            purge_end="2022-10-07",
            validation_start="2022-10-14",
            validation_end="2022-11-18",
        )]
        assignments, metadata = build_frozen_assignments(self.panel(), folds, 12, 2)
        train = assignments[assignments.split.eq("train")]
        validation = assignments[assignments.split.eq("validation")]
        self.assertLess(train.target_date.max(), validation.trade_date.min())
        self.assertEqual(metadata[0]["validation_stock_count"], 2)
        self.assertIn("validation_sample_content_sha256", metadata[0])


if __name__ == "__main__":
    unittest.main()
