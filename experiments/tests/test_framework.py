from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import DataBundle, evaluate_predictions, load_config, prediction_frame
from experiments.models import ArimaBaseline, MovingAverageBaseline, NaiveBaseline, build_model


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(REPO_ROOT / "experiments/configs/stage_b_baselines.json", REPO_ROOT)
        cls.data = DataBundle.load(Path(cls.config["data_root"]))

    def test_stage_a_split_counts(self) -> None:
        self.assertEqual({k: len(v) for k, v in self.data.samples.items()}, {
            "train": 688, "validation": 120, "test": 210,
        })

    def test_naive_outputs_every_test_sample(self) -> None:
        prediction = NaiveBaseline().predict(self.data, "test")
        self.assertEqual(len(prediction), 210)
        self.assertTrue(np.all(prediction == 0.0))

    def test_moving_average_is_past_only(self) -> None:
        model = MovingAverageBaseline(window=4)
        prediction = model.predict(self.data, "validation")
        first = self.data.samples["validation"].iloc[0]
        history = self.data.panel[
            (self.data.panel.stock_code == first.stock_code)
            & (self.data.panel.trade_date <= first.trade_date)
        ].tail(4)
        self.assertAlmostEqual(prediction[0], history.return_1w.mean(), places=12)

    def test_metrics_and_prediction_schema(self) -> None:
        samples = self.data.samples["validation"].head(3)
        frame = prediction_frame(samples, np.zeros(3), "validation")
        metrics = evaluate_predictions(frame)
        self.assertEqual(metrics["aggregate"]["samples"], 3)
        self.assertIn("predicted_close", frame.columns)

    def test_closed_form_ar1_forecast(self) -> None:
        series = np.asarray([1.0, 3.0, 7.0, 15.0, 31.0])
        self.assertAlmostEqual(ArimaBaseline._closed_form_ar1_forecast(series), 63.0, places=10)

    def test_minimalist_transformer_configuration(self) -> None:
        model = build_model("minimalist_transformer", self.config, 20260723)
        self.assertEqual(model.name, "minimalist_transformer")
        self.assertEqual(model.config["d_model"] % model.config["nhead"], 0)
        self.assertEqual(model.config["num_layers"], 1)



if __name__ == "__main__":
    unittest.main()
