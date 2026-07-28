from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from stage_f.gan import (
    BoundedConditionalGenerator,
    SpectralTemporalCritic,
    critic_wgan_gp_loss,
    deterministic_noise,
    generator_adversarial_loss,
    gradient_penalty,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[2]


class F2GanContractTest(unittest.TestCase):
    def test_generator_is_deterministic_bounded_and_shape_preserving(self) -> None:
        torch.manual_seed(7)
        generator = BoundedConditionalGenerator()
        values = torch.randn(2, 8, 5, 6)
        available = torch.ones(2, 5, dtype=torch.bool)
        available[:, -1] = False
        left_noise = deterministic_noise((2, 8, 5, 8), 20260725)
        right_noise = deterministic_noise((2, 8, 5, 8), 20260725)
        left, left_delta = generator(values, left_noise, available)
        right, right_delta = generator(values, right_noise, available)
        self.assertEqual(left.shape, values.shape)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        torch.testing.assert_close(left_delta, right_delta, rtol=0.0, atol=0.0)
        self.assertLessEqual(float(left_delta.detach().abs().max()), 0.05 + 1e-7)
        self.assertEqual(float(left_delta.detach()[:, :, -1].abs().max()), 0.0)

    def test_critic_and_losses_have_finite_gradients_without_optimizer_step(self) -> None:
        torch.manual_seed(11)
        generator = BoundedConditionalGenerator()
        critic = SpectralTemporalCritic()
        values = torch.randn(2, 8, 5, 6)
        noise = deterministic_noise((2, 8, 5, 8), 91)
        fake, delta = generator(values, noise)
        real_score = critic(values)
        fake_score = critic(fake.detach())
        interpolation = torch.full((2, 1, 1, 1), 0.5)
        penalty = gradient_penalty(critic, values, fake.detach(), interpolation)
        critic_loss = critic_wgan_gp_loss(real_score, fake_score, penalty)
        self.assertTrue(torch.isfinite(critic_loss))
        critic_loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in critic.parameters()))
        detail = generator_adversarial_loss(
            critic(fake),
            torch.tensor([0.02, 0.03]),
            torch.tensor([0.04, 0.05]),
            delta,
        )
        self.assertTrue(torch.isfinite(detail.total))
        detail.total.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in generator.parameters()))

    def test_parameter_bounds_and_training_authorization_boundary(self) -> None:
        self.assertLess(parameter_count(BoundedConditionalGenerator()), 100000)
        self.assertLess(parameter_count(SpectralTemporalCritic()), 100000)
        config_path = ROOT / "stage_f/configs/f2_gan_addendum_v1.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertFalse(config["authorization"]["gan_training_authorized"])
            self.assertTrue(config["authorization"]["explicit_training_authorization_required"])


if __name__ == "__main__":
    unittest.main()
