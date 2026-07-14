import unittest

import torch

from src.modeling.critic import (
    LatentCritic,
    LatentGenerator,
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
                    self.assertEqual(scores.shape, torch.Size([2, 1]))

    def test_critic_rejects_too_small_slices(self):
        with self.assertRaisesRegex(ValueError, "at least 16"):
            LatentCritic(2)(torch.zeros(1, 2, 15, 15))

    def test_critic_is_invariant_to_channel_affine_statistics(self):
        critic = LatentCritic(2, base_ch=4)
        latent = torch.randn(3, 2, 16, 16)

        original = critic(latent)
        shifted = critic(latent * 1.5 + 0.25)

        self.assertTrue(torch.allclose(original, shifted, atol=1e-5, rtol=1e-5))

    def test_critic_does_not_amplify_nearly_constant_channels(self):
        torch.manual_seed(0)
        critic = LatentCritic(8, base_ch=4)
        latent = torch.randn(3, 8, 16, 16) * 1e-6
        latent[:, -1] = torch.randn(3, 16, 16)
        perturbed = latent.clone()
        perturbed[:, :-1] = torch.randn_like(perturbed[:, :-1]) * 1e-6

        original = critic(latent)
        changed = critic(perturbed)

        self.assertTrue(torch.allclose(original, changed, atol=1e-4, rtol=1e-4))

    def test_balanced_sampler_uses_all_axes(self):
        volume = torch.zeros(1, 1, 16, 16, 16)
        volume[:, :, 1:] = 1.0

        samples = sample_slices(volume, count=3, crop_size=16)

        self.assertEqual(samples.shape, torch.Size([3, 1, 16, 16]))

    def test_single_slice_sampler_cycles_axes_with_step_offset(self):
        depth = torch.arange(16).view(16, 1, 1) * 10_000
        row = torch.arange(16).view(1, 16, 1) * 100
        column = torch.arange(16).view(1, 1, 16)
        volume = (depth + row + column).float().view(1, 1, 16, 16, 16)

        increments = []
        for step in range(6):
            sampled = sample_slices(
                volume,
                count=1,
                crop_size=16,
                axis_offset=step % 3,
            )[0, 0]
            increments.append(
                (
                    int(sampled[1, 0] - sampled[0, 0]),
                    int(sampled[0, 1] - sampled[0, 0]),
                )
            )

        self.assertEqual(
            increments,
            [
                (100, 1),
                (10_000, 1),
                (10_000, 100),
                (100, 1),
                (10_000, 1),
                (10_000, 100),
            ],
        )

    def test_objectives_have_expected_signs(self):
        real = torch.tensor([2.0])
        fake = torch.tensor([-1.0])
        penalty = torch.tensor(0.5)

        critic = critic_loss(real, fake, penalty, gp_weight=2.0)
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

    def test_generator_creates_latent_patches_from_noise(self):
        for size in (16, 20, 32):
            with self.subTest(size=size):
                generator = LatentGenerator(
                    latent_ch=4,
                    latent_size=size,
                    noise_ch=8,
                    base_ch=8,
                )
                generated = generator(torch.randn(2, 8))

                self.assertEqual(generated.shape, torch.Size([2, 4, size, size]))
                self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
