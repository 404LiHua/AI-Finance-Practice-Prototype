from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from stage_d.custody import DataCustodyGuard, DataCustodyViolation
from stage_d.d2_baselines import (
    apply_fixed_shrinkage,
    build_fold_bundle,
    load_locked_config,
    registered_models,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "stage_d/configs/d2_baselines.json"


class D2BaselinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_locked_config(CONFIG_PATH, REPO_ROOT)
        cls.protocol = validate_protocol(cls.config)

    def test_preregistered_grid_is_exact(self) -> None:
        self.assertEqual(len(registered_models(self.config)), 17)
        self.assertEqual(self.config["seeds"], [20260723, 20260724, 20260725])
        self.assertEqual(self.config["shrinkage"]["alphas"], [0.25, 0.5, 0.75])

    def test_fixed_shrinkage_rejects_unregistered_alpha(self) -> None:
        values = np.asarray([-0.1, 0.2])
        np.testing.assert_allclose(apply_fixed_shrinkage(values, 0.5), [-0.05, 0.1])
        with self.assertRaises(ValueError):
            apply_fixed_shrinkage(values, 0.6)

    def test_each_fold_bundle_matches_registered_hashes(self) -> None:
        for fold in self.protocol["folds"]:
            bundle, evidence = build_fold_bundle(
                self.config, self.protocol, fold["fold_id"], REPO_ROOT,
            )
            self.assertEqual(evidence["train_row_set_sha256"], fold["train_row_set_sha256"])
            self.assertEqual(evidence["validation_row_set_sha256"], fold["validation_row_set_sha256"])
            self.assertEqual(set(bundle.samples), {"train", "validation"})
            self.assertLessEqual(bundle.panel["trade_date"].max().date().isoformat(), fold["validation_end"])

    def test_custody_guard_blocks_c4_and_future_screening_paths(self) -> None:
        guard = DataCustodyGuard.from_config(
            Path(self.config["custody_config_path"]), REPO_ROOT,
        )
        policy = json.loads(Path(self.config["custody_config_path"]).read_text(encoding="utf-8"))
        with self.assertRaises(DataCustodyViolation):
            guard.assert_path_allowed(REPO_ROOT / policy["repo_root_relative_forbidden_paths"][0])
        with self.assertRaises(DataCustodyViolation):
            guard.assert_path_allowed(REPO_ROOT / "data/future_d_screening_predictions.csv")


if __name__ == "__main__":
    unittest.main()
