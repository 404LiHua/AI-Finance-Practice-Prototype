import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.core import DataBundle
from stage_c.inference import FrozenSequenceBuilder, LoadedFixedEnsemble, LoadedStageCComponent


class FrozenSequenceBuilderTest(unittest.TestCase):
    def test_rebuilds_left_padded_sequences(self) -> None:
        panel = pd.DataFrame({
            "stock_code": ["A", "A", "A"],
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "split": ["train", "validation", "validation"],
            "sample_eligible": [True, True, True],
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
        })
        data = DataBundle(
            panel=panel,
            samples={
                "train": panel.iloc[[0]].copy(),
                "validation": panel.iloc[[1, 2]].copy(),
                "test": panel.iloc[0:0].copy(),
            },
            feature_columns=["f1", "f2"],
        )
        builder = FrozenSequenceBuilder(
            feature_columns=["f1", "f2"],
            medians=np.array([0.0, 0.0]),
            means=np.array([0.0, 0.0]),
            stds=np.array([1.0, 10.0]),
            sequence_length=3,
        )
        values = builder.build(data, "validation")
        self.assertEqual(tuple(values.shape), (2, 3, 2))
        np.testing.assert_allclose(values[0], [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_allclose(values[1], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    def test_rejects_feature_order_mismatch(self) -> None:
        panel = pd.DataFrame({
            "stock_code": ["A"], "trade_date": pd.to_datetime(["2026-01-02"]),
            "split": ["validation"], "sample_eligible": [True], "f1": [1.0], "f2": [2.0],
        })
        data = DataBundle(panel, {"train": panel.iloc[0:0], "validation": panel, "test": panel.iloc[0:0]}, ["f2", "f1"])
        builder = FrozenSequenceBuilder(["f1", "f2"], np.zeros(2), np.zeros(2), np.ones(2), 1)
        with self.assertRaises(ValueError):
            builder.build(data, "validation")

    def test_missing_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            LoadedStageCComponent(Path("definitely_missing_stage_c_model.pt"))

    def test_malformed_manifest_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('{"components": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                LoadedFixedEnsemble(manifest, root)


if __name__ == "__main__":
    unittest.main()
