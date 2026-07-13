import unittest
from unittest.mock import patch

import torch

from src.app.api.options import SliceGANTrainConfig
from src.modeling.slicegan import SliceGANCritic, SliceGANGenerator
from src.pipelines.training.slicegan import (
    _sample_real,
    train_step,
)


class SliceGANTrainingTest(unittest.TestCase):
    def test_real_sampling_uses_selected_reference_pool(self):
        anchors = torch.ones(2, 2, 16, 16)
        diffusion = torch.full_like(anchors, 2.0)

        anchor_batch = _sample_real(
            anchors,
            diffusion,
            batch_size=4,
            mix_probability=0.0,
        )
        diffusion_batch = _sample_real(
            anchors,
            diffusion,
            batch_size=4,
            mix_probability=1.0,
        )

        self.assertTrue(torch.equal(anchor_batch, torch.ones_like(anchor_batch)))
        self.assertTrue(torch.equal(diffusion_batch, torch.full_like(diffusion_batch, 2.0)))

    def test_real_sampling_augmentation_only_reorders_values(self):
        image = torch.arange(16 * 16).reshape(1, 1, 16, 16).float()

        sampled = _sample_real(
            image,
            torch.zeros_like(image),
            batch_size=1,
            mix_probability=0.0,
        )

        self.assertTrue(torch.equal(sampled.flatten().sort().values, image.flatten()))

    def test_train_step_updates_both_models_and_restores_critic_gradients(self):
        generator = SliceGANGenerator(latent_ch=2, base_ch=8)
        critic = SliceGANCritic(latent_ch=2, base_ch=4)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
        optimizer_d = torch.optim.Adam(critic.parameters(), lr=1e-3)
        anchors = torch.randn(2, 2, 16, 16)
        diffusion = torch.randn_like(anchors)
        generator_before = [value.detach().clone() for value in generator.parameters()]
        critic_before = [value.detach().clone() for value in critic.parameters()]
        config = SliceGANTrainConfig(
            steps=1,
            mix_steps=0,
            critic_steps=2,
            batch_size=2,
            reference_count=1,
        )

        with patch(
            "src.pipelines.training.slicegan.gradient_penalty",
            wraps=lambda model, real, fake: (model(real).mean() * 0.0 + 1.0),
        ) as penalty:
            stats = train_step(
                generator,
                critic,
                optimizer_g,
                optimizer_d,
                anchors,
                diffusion,
                noise_size=4,
                mixed=True,
                config=config,
            )

        self.assertEqual(penalty.call_count, 2)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(generator_before, generator.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(critic_before, critic.parameters())
            )
        )
        self.assertTrue(all(parameter.requires_grad for parameter in critic.parameters()))
        self.assertEqual(
            set(stats),
            {
                "slicegan_critic_margin",
                "slicegan_generator_loss",
                "slicegan_gradient_penalty",
            },
        )
