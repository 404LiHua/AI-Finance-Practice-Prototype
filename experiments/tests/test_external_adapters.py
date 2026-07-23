from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import DataBundle, load_config
from experiments.run_external_baselines import build_adapter


class ExternalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(REPO_ROOT / "experiments/configs/external_adapters.json", REPO_ROOT)
        cls.data = DataBundle.load(Path(cls.config["data_root"]))

    def test_upstream_files_exist(self) -> None:
        for name in ("frets", "timegnn"):
            adapter = build_adapter(name, self.config, 20260723)
            self.assertTrue(adapter.source_file.is_file())

    def test_adapter_features(self) -> None:
        frets = build_adapter("frets", self.config, 20260723)
        timegnn = build_adapter("timegnn", self.config, 20260723)
        self.assertEqual(frets.select_features(self.data), ["return_1w"])
        self.assertEqual(timegnn.select_features(self.data), self.data.feature_columns)

    def test_identical_sample_counts_and_sequence_alignment(self) -> None:
        adapter = build_adapter("frets", self.config, 20260723)
        adapter._prepare_scaler(self.data)
        for split, expected in (("train", 688), ("validation", 120), ("test", 210)):
            x, y = adapter._sequences(self.data, split)
            self.assertEqual(x.shape, (expected, 8, 1))
            self.assertEqual(y.shape, (expected,))


if __name__ == "__main__":
    unittest.main()
