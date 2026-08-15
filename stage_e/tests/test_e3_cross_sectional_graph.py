from __future__ import annotations

import unittest

import numpy as np
import torch

from stage_e.build_e3_graphs import build_industry_graph, correlation_topk
from stage_e.models.cross_sectional_graph import CrossSectionalGraphBlock, CrossSectionalGraphLearner


class E3CrossSectionalGraphTests(unittest.TestCase):
    def test_industry_graph_is_row_stochastic(self) -> None:
        graph = build_industry_graph(["金融", "金融", "其他"], {"unknown_groups": ["其他"], "unknown_self_only": True})
        np.testing.assert_allclose(graph.sum(axis=1), 1.0)
        self.assertEqual(float(graph[2, 2]), 1.0)
        self.assertEqual(float(graph[2, 0]), 0.0)

    def test_correlation_graph_is_bounded_topk(self) -> None:
        correlation = np.asarray([[1.0, 0.9, -0.8], [0.9, 1.0, 0.2], [-0.8, 0.2, 1.0]])
        graph = correlation_topk(correlation, top_k=1, minimum=0.05, self_weight=1.0)
        np.testing.assert_allclose(graph.sum(axis=1), 1.0)
        self.assertTrue(((graph > 0).sum(axis=1) <= 2).all())

    def test_adaptive_graph_shape_determinism_and_gradient(self) -> None:
        torch.manual_seed(7)
        learner = CrossSectionalGraphLearner(hidden_dim=6, top_k=3, sampling_mode="deterministic")
        values = torch.randn(2, 8, 6, requires_grad=True)
        first = learner(values)
        second = learner(values)
        self.assertEqual(tuple(first.shape), (2, 8, 8))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.sum(dim=-1), torch.ones(2, 8), atol=1e-6))
        self.assertTrue(bool(((first > 0).sum(dim=-1) <= 4).all()))
        first.sum().backward()
        self.assertIsNotNone(values.grad)

    def test_permutation_equivariance(self) -> None:
        torch.manual_seed(11)
        learner = CrossSectionalGraphLearner(hidden_dim=4, top_k=2, sampling_mode="deterministic").eval()
        values = torch.randn(1, 6, 4)
        permutation = torch.tensor([2, 5, 1, 0, 4, 3])
        original = learner(values)
        permuted = learner(values[:, permutation])
        expected = original[:, permutation][:, :, permutation]
        self.assertTrue(torch.allclose(permuted, expected, atol=1e-6))

    def test_unavailable_node_falls_back_to_self(self) -> None:
        learner = CrossSectionalGraphLearner(hidden_dim=4, top_k=2)
        values = torch.randn(1, 4, 4)
        available = torch.tensor([[True, False, True, True]])
        graph = learner(values, available)
        self.assertEqual(float(graph[0, 1, 1].detach()), 1.0)
        self.assertEqual(float(graph[0, 0, 1].detach()), 0.0)

    def test_graph_block_preserves_shape(self) -> None:
        block = CrossSectionalGraphBlock(hidden_dim=5)
        values = torch.randn(2, 7, 5)
        adjacency = torch.eye(7).unsqueeze(0).repeat(2, 1, 1)
        self.assertEqual(tuple(block(values, adjacency).shape), (2, 7, 5))

    def test_gumbel_is_training_only_and_temperature_is_bounded(self) -> None:
        torch.manual_seed(13)
        learner = CrossSectionalGraphLearner(hidden_dim=4, top_k=2, sampling_mode="gumbel")
        learner.set_temperature(0.3)
        self.assertEqual(learner.temperature, 0.3)
        with self.assertRaises(ValueError):
            learner.set_temperature(0.0)
        values = torch.randn(1, 6, 4)
        learner.eval()
        first = learner(values)
        second = learner(values)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
