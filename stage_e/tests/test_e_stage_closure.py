from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "releases/e_stage_best_model_v1"


class EStageClosureTest(unittest.TestCase):
    def test_release_contains_frozen_unique_candidate(self) -> None:
        metrics = json.loads((RELEASE / "model_metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["model_id"], "stock_node_gwnet_fixed_industry_l8")
        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["failed_gate_count"], 0)
        self.assertEqual(len(list((RELEASE / "checkpoints").rglob("*.pt"))), 9)
        self.assertGreater(manifest["artifact_count"], 30)

    def test_standalone_three_seed_inference_shape(self) -> None:
        spec = importlib.util.spec_from_file_location("e_stage_release_inference", RELEASE / "inference.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        paths = [RELEASE / "checkpoints/E_RO_03" / f"seed_{seed}.pt" for seed in (20260723, 20260724, 20260725)]
        prediction, seeds = module.predict(np.zeros((1, 8, 100, 6), dtype=np.float32), paths)
        self.assertEqual(prediction.shape, (1, 100))
        self.assertEqual(seeds.shape, (3, 1, 100))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
