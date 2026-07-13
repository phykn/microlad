import unittest

import torch

from src.modeling.critic import (
    LatentCritic,
    critic_loss,
    gradient_penalty,
    guidance_loss,
    sample_slices,
)


class LatentCriticTest(unittest.TestCase):
    def test_critic_supports_trained_vae_latent_sizes(self):
        with torch.device("meta"):
            critic = LatentCritic(latent_ch=8)
            for size in (16, 20, 24, 32):
                with self.subTest(size=size):
                    scores = critic(torch.empty(2, 8, size, size, device="meta"))
                    self.assertEqual(scores.shape[0], 2)
                    self.assertGreater(scores.shape[-1], 0)

    def test_critic_rejects_too_small_slices(self):
        with self.assertRaisesRegex(ValueError, "at least 16"):
            LatentCritic(2)(torch.zeros(1, 2, 15, 15))

    def test_balanced_sampler_uses_all_axes(self):
        volume = torch.zeros(1, 1, 16, 16, 16)
        volume[:, :, 1:] = 1.0

        samples = sample_slices(volume, count=3, crop_size=16)

        self.assertEqual(samples.shape, torch.Size([3, 1, 16, 16]))

    def test_objectives_have_expected_signs(self):
        real = torch.tensor([2.0])
        fake = torch.tensor([-1.0])
        penalty = torch.tensor(0.5)

        critic = critic_loss(real, fake, penalty, gradient_weight=2.0)
        guidance = guidance_loss(fake)

        self.assertEqual(float(critic), -2.0)
        self.assertEqual(float(guidance), 1.0)

    def test_gradient_penalty_is_finite_and_differentiable(self):
        critic = LatentCritic(2, base_ch=4)
        real = torch.randn(2, 2, 16, 16)
        fake = torch.randn_like(real)

        penalty = gradient_penalty(critic, real, fake)
        penalty.backward()

        self.assertTrue(torch.isfinite(penalty))
        self.assertTrue(any(parameter.grad is not None for parameter in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
