import unittest

import torch

from stage_c.models import GraphFrequencyModel


class GraphFrequencyModelTest(unittest.TestCase):
    def test_forward_shapes_and_backward(self) -> None:
        torch.manual_seed(13)
        model = GraphFrequencyModel(
            input_dim=14, sequence_length=8, hidden_dim=16, nhead=4,
            dim_feedforward=32, top_k=2, dropout=0.0,
        )
        values = torch.randn(5, 8, 14)
        details = model(values, return_details=True)
        self.assertEqual(tuple(details["prediction"].shape), (5,))
        self.assertEqual(tuple(details["adjacency"].shape), (5, 8, 8))
        self.assertEqual(tuple(details["gate"].shape), (5, 16))
        self.assertTrue(torch.isfinite(details["prediction"]).all())
        details["prediction"].square().mean().backward()
        self.assertIsNotNone(model.head.weight.grad)

    def test_ablation_modes(self) -> None:
        values = torch.randn(3, 8, 6)
        variants = (
            {"graph_mode": "identity"},
            {"graph_mode": "temporal_neighbor"},
            {"graph_mode": "identity", "use_frequency": False},
            {"fusion_mode": "mean"},
            {"top_k": 4},
            {"graph_mode": "temporal_neighbor", "use_frequency": False, "use_time_graph": True, "fusion_mode": "mean"},
        )
        for overrides in variants:
            with self.subTest(overrides=overrides):
                model = GraphFrequencyModel(
                    input_dim=6, sequence_length=8, hidden_dim=16, nhead=4,
                    dim_feedforward=32, dropout=0.0, **overrides,
                )
                details = model(values, return_details=True)
                self.assertEqual(tuple(details["prediction"].shape), (3,))
                self.assertEqual(tuple(details["adjacency"].shape), (3, 8, 8))
                self.assertTrue(torch.isfinite(details["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
