from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.bounded_ablations import (
    PRICE_FEATURES,
    FreTSBoundedAblationAdapter,
    minimalist_feature_view,
)
from experiments.core import DataBundle


REPO_ROOT = Path(__file__).resolve().parents[2]


class BoundedAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        columns = PRICE_FEATURES + ["text_count", "text_svd_01"]
        panel = pd.DataFrame({column: [0.0] for column in columns})
        panel["stock_code"] = ["000001.SZ"]
        panel["trade_date"] = pd.to_datetime(["2023-01-06"])
        samples = {split: panel.copy() for split in ("train", "validation", "test")}
        self.data = DataBundle(panel=panel, samples=samples, feature_columns=columns)

    def test_minimalist_feature_views_are_nested(self) -> None:
        price = minimalist_feature_view(self.data, "price_only")
        price_text = minimalist_feature_view(self.data, "price_text")
        self.assertEqual(PRICE_FEATURES, price.feature_columns)
        self.assertEqual(PRICE_FEATURES + ["text_count", "text_svd_01"], price_text.feature_columns)

    def test_frets_return_channel_is_first(self) -> None:
        adapter = FreTSBoundedAblationAdapter(
            {"feature_set": "return_ohlc"}, {"root": "."}, 1, "variant",
        )
        features = adapter.select_features(self.data)
        self.assertEqual("return_1w", features[0])
        self.assertEqual(5, len(features))

    def test_declared_configuration_budget(self) -> None:
        config = json.loads(
            (REPO_ROOT / "experiments/configs/bounded_ablations.json").read_text(encoding="utf-8")
        )
        frets = [v for v in config["variants"].values() if v["family"] == "frets"]
        transformer = [
            v for v in config["variants"].values()
            if v["family"] == "minimalist_transformer"
        ]
        self.assertEqual(4, len(frets))
        self.assertEqual(2, len(transformer))
        self.assertEqual("SELECTION_EXPOSED_TRAIN_NON_INDEPENDENT", config["evidence_class"])


if __name__ == "__main__":
    unittest.main()
