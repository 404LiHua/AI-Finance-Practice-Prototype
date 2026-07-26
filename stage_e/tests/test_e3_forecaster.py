from __future__ import annotations

import unittest

import torch

from stage_e.models.cross_sectional_forecaster import CrossSectionalTemporalForecaster


class E3ForecasterTests(unittest.TestCase):
    def test_forward_shapes_and_details(self) -> None:
        model = CrossSectionalTemporalForecaster(input_dim=6, stock_count=12, hidden_dim=16, top_k=4)
        values = torch.randn(3, 8, 12, 6)
        details = model(values, return_details=True)
        self.assertEqual(tuple(details["prediction"].shape), (3, 12))
        self.assertEqual(tuple(details["adjacency"].shape), (3, 12, 12))
        self.assertTrue(torch.allclose(details["adjacency"].sum(dim=-1), torch.ones(3, 12), atol=1e-6))

    def test_wrong_stock_count_is_rejected(self) -> None:
        model = CrossSectionalTemporalForecaster(input_dim=3, stock_count=5, hidden_dim=8, top_k=2)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 6, 3))


if __name__ == "__main__":
    unittest.main()
