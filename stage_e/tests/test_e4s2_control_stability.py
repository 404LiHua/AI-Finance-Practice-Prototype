from __future__ import annotations

import unittest

import torch
from torch import nn

from stage_e.run_e4s2_control_stability import update_ema


class E4S2ControlStabilityTest(unittest.TestCase):
    def test_ema_updates_floating_parameters(self) -> None:
        model = nn.Linear(2, 1)
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        before = ema["weight"].clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        update_ema(ema, model, 0.5)
        self.assertTrue(torch.allclose(ema["weight"], before + 0.5))


if __name__ == "__main__":
    unittest.main()
