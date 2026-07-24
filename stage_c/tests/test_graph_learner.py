import unittest

import torch

from stage_c.models.graph_learner import DynamicGraphLearner


class DynamicGraphLearnerTest(unittest.TestCase):
    def test_sparse_row_stochastic_graph_and_gradient(self) -> None:
        torch.manual_seed(7)
        learner = DynamicGraphLearner(hidden_dim=8, top_k=2, temperature=0.8)
        values = torch.randn(4, 6, 8, requires_grad=True)
        adjacency = learner(values)
        self.assertEqual(tuple(adjacency.shape), (4, 6, 6))
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertTrue(torch.allclose(adjacency.sum(-1), torch.ones(4, 6), atol=1e-6))
        self.assertLessEqual(int((adjacency > 0).sum(-1).max()), 2)
        adjacency.square().sum().backward()
        self.assertIsNotNone(learner.query.weight.grad)
        self.assertGreater(float(learner.query.weight.grad.abs().sum()), 0.0)

    def test_deterministic_mode_is_repeatable_during_training(self) -> None:
        torch.manual_seed(9)
        learner = DynamicGraphLearner(
            hidden_dim=8, top_k=2, temperature=0.8, sampling_mode="deterministic",
        )
        learner.train()
        values = torch.randn(2, 6, 8)
        first = learner(values)
        second = learner(values)
        self.assertTrue(torch.equal(first, second))
        learner.set_temperature(0.4)
        third = learner(values)
        self.assertFalse(torch.equal(first, third))


if __name__ == "__main__":
    unittest.main()
