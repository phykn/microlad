import unittest

import torch

from src.modeling.gan import (
    ImageCritic,
    LatentGenerator,
    critic_loss,
    gradient_penalty,
    guidance_loss,
    morphology_feature_loss,
)


class ImageCriticTest(unittest.TestCase):
    def test_critic_supports_decoded_image_sizes(self):
        with torch.device("meta"):
            for size in (16, 32, 64, 128):
                with self.subTest(size=size):
                    critic = ImageCritic(num_phases=3, image_size=size)
                    scores = critic(torch.empty(2, 3, size, size, device="meta"))
                    self.assertEqual(scores.shape, torch.Size([2, 1]))

    def test_critic_rejects_too_small_images(self):
        with self.assertRaisesRegex(ValueError, "at least 16"):
            ImageCritic(2, image_size=15)

    def test_critic_rejects_images_that_do_not_match_vae_size(self):
        critic = ImageCritic(2, image_size=64, base_ch=4)

        with self.assertRaisesRegex(ValueError, "configured VAE image size"):
            critic(torch.zeros(1, 2, 128, 128))

    def test_objectives_have_expected_signs(self):
        real = torch.tensor([2.0])
        fake = torch.tensor([-1.0])
        penalty = torch.tensor(0.5)

        critic = critic_loss(real, fake, penalty, gp_weight=2.0)
        guidance = guidance_loss(fake)

        self.assertEqual(float(critic), -2.0)
        self.assertEqual(float(guidance), 1.0)

    def test_morphology_feature_loss_matches_unpaired_feature_statistics(self):
        critic = ImageCritic(3, image_size=16, base_ch=4)
        references = torch.randn(3, 3, 16, 16)
        generated = references.flip(0).clone().requires_grad_()

        loss = morphology_feature_loss(critic, generated, references)
        loss.backward()

        self.assertLess(float(loss.detach()), 1e-10)
        self.assertIsNotNone(generated.grad)
        self.assertTrue(torch.isfinite(generated.grad).all())

    def test_gradient_penalty_is_finite_and_differentiable(self):
        critic = ImageCritic(3, image_size=16, base_ch=4)
        real = torch.randn(2, 3, 16, 16).softmax(dim=1)
        fake = torch.randn_like(real).softmax(dim=1)

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
