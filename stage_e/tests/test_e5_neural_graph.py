from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage_e.e5.neural_graph import (
    FixedIndustryGraphWaveNet, LSTMNetwork, StableTimeGNNNetwork, TCNNetwork,
    fixed_industry_adjacency,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class E5NeuralGraphTest(unittest.TestCase):
    def test_network_output_shapes(self) -> None:
        values = torch.randn(4, 8, 6)
        lstm = LSTMNetwork(6, {"hidden_size": 8, "num_layers": 1, "dropout": 0.0})
        tcn = TCNNetwork(6, {"hidden_channels": 8, "kernel_size": 3, "dilations": [1, 2], "dropout": 0.0})
        timegnn = StableTimeGNNNetwork(6, 8, {"hidden_dim": 8, "top_k": 2, "temperature": 0.5, "graph_layers": 1})
        self.assertEqual(tuple(lstm(values).shape), (4,))
        self.assertEqual(tuple(tcn(values).shape), (4,))
        self.assertEqual(tuple(timegnn(values).shape), (4,))

    def test_timegnn_adjacency_is_deterministic_and_topk(self) -> None:
        model = StableTimeGNNNetwork(6, 8, {"hidden_dim": 8, "top_k": 2, "temperature": 0.5, "graph_layers": 1})
        values = torch.randn(3, 8, 6)
        with torch.no_grad():
            _, left = model(values, return_adjacency=True)
            _, right = model(values, return_adjacency=True)
        torch.testing.assert_close(left, right)
        self.assertTrue(torch.equal((left > 0).sum(dim=-1), torch.full((3, 8), 2)))

    def test_fixed_industry_graph_uses_real_stock_order(self) -> None:
        stocks = np.array(["a", "b", "c"])
        universe = pd.DataFrame({"stock_code": stocks, "industry_group": ["x", "x", "y"]})
        adjacency, industries = fixed_industry_adjacency(stocks, universe)
        self.assertEqual(industries, ["x", "x", "y"])
        np.testing.assert_allclose(adjacency.sum(axis=1), 1.0)
        self.assertGreater(adjacency[0, 1], 0.0)
        self.assertEqual(adjacency[0, 2], 0.0)
        model = FixedIndustryGraphWaveNet(6, adjacency, {
            "hidden_channels": 8, "kernel_size": 3, "dilations": [1, 2],
            "graph_order": 2, "dropout": 0.0,
        })
        self.assertEqual(tuple(model(torch.randn(2, 8, 3, 6)).shape), (2, 3))

    def test_locked_collection_and_graph_semantics(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e5_neural_graph_baselines_v1.json").read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in config["models"]], [
            "lstm_price_l8", "tcn_price_l8", "timegnn_deterministic_topk_l8",
            "stock_node_gwnet_fixed_industry_l8",
        ])
        self.assertEqual(config["models"][2]["stabilization"], "deterministic_temperature_scaled_topk_no_gumbel")
        self.assertEqual(config["models"][3]["node_semantics"], "real_stock_code_nodes")
        self.assertFalse(any(config["restrictions"].values()))

    def test_three_seed_review_only_adds_two_frozen_seeds(self) -> None:
        import hashlib
        base_path = REPO_ROOT / "stage_e/configs/e5_neural_graph_baselines_v1.json"
        review = json.loads((REPO_ROOT / "stage_e/configs/e5_neural_graph_three_seed_review_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(base_path.read_bytes()).hexdigest(), review["base_protocol_config_sha256"])
        self.assertEqual(review["new_training_seeds"], [20260723, 20260724])
        self.assertEqual(review["reused_engineering_seed"], 20260725)
        self.assertFalse(any(review["restrictions"].values()))


if __name__ == "__main__":
    unittest.main()
