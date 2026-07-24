import unittest

import torch
from torch import nn

from stage_c.models import GraphFrequencyModel


class TinyOverfitTest(unittest.TestCase):
    def test_tiny_batch_loss_decreases(self) -> None:
        torch.manual_seed(17)
        values = torch.randn(12, 6, 5)
        target = 0.25 * values[:, -1, 0] - 0.15 * values[:, -2, 1]
        model = GraphFrequencyModel(
            input_dim=5, sequence_length=6, hidden_dim=12, nhead=3,
            dim_feedforward=24, top_k=2, dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.eval()
        with torch.no_grad():
            initial = float(loss_fn(model(values), target))
        model.train()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(values), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = float(loss_fn(model(values), target))
        self.assertLess(final, initial * 0.35)


if __name__ == "__main__":
    unittest.main()

