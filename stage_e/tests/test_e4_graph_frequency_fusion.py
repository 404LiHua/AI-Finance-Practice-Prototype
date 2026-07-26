from __future__ import annotations

import unittest

import torch

from stage_e.models.graph_frequency_fusion import (
    CrossSectionalFrequencyGraphBlock,
    CrossSectionalTimeGraphBlock,
    GraphFrequencyFusionModel,
)


class E4GraphFrequencyFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.values = torch.randn(2, 12, 10, 6)
        self.identity = torch.eye(10).unsqueeze(0).repeat(2, 1, 1)

    def test_time_graph_shape_and_gradient(self) -> None:
        values = self.values.clone().requires_grad_(True)
        output = CrossSectionalTimeGraphBlock(6)(values, self.identity)
        self.assertEqual(tuple(output.shape), tuple(values.shape))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_frequency_graph_uses_time_fft_and_preserves_shape(self) -> None:
        values = self.values.clone().requires_grad_(True)
        output = CrossSectionalFrequencyGraphBlock(6)(values, self.identity)
        self.assertEqual(tuple(output.shape), tuple(values.shape))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_all_branch_and_fusion_modes(self) -> None:
        for branch in ("temporal_only", "time_graph", "frequency_graph", "dual_branch"):
            for fusion in ("concat", "fixed_mean", "gated", "residual"):
                model = GraphFrequencyFusionModel(
                    input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
                    graph_mode="learned_deterministic", branch_mode=branch, fusion_mode=fusion,
                )
                details = model(self.values, return_details=True)
                self.assertEqual(tuple(details["prediction"].shape), (2, 10))
                self.assertEqual(tuple(details["adjacency"].shape), (2, 10, 10))
                self.assertTrue(torch.isfinite(details["prediction"]).all())

    def test_provided_adjacency_is_used_without_replacement(self) -> None:
        model = GraphFrequencyFusionModel(
            input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
            graph_mode="provided", branch_mode="time_graph",
        )
        details = model(self.values, adjacency=self.identity, return_details=True)
        self.assertTrue(torch.equal(details["adjacency"], self.identity))

    def test_early_and_mid_text_fusion(self) -> None:
        text = torch.randn(2, 10, 7)
        for mode in ("early", "mid"):
            model = GraphFrequencyFusionModel(
                input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
                graph_mode="learned_deterministic", branch_mode="dual_branch",
                fusion_mode="fixed_mean", text_dim=7, text_fusion=mode,
            )
            prediction = model(self.values, text_features=text)
            self.assertEqual(tuple(prediction.shape), (2, 10))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_missing_text_is_rejected_when_enabled(self) -> None:
        model = GraphFrequencyFusionModel(
            input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
            text_dim=5, text_fusion="early",
        )
        with self.assertRaises(ValueError):
            model(self.values)


if __name__ == "__main__":
    unittest.main()
