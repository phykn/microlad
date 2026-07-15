import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.modeling.gan import (
    ImageCritic,
    LatentGenerator,
    critic_loss,
    gradient_penalty,
)
from src.pipeline.predict.reconstruction.volume import decode_volume_probs
from src.pipeline.train import GANTrainer


class TinyVAE(torch.nn.Module):
    image_size = 16
    latent_size = 16
    latent_ch = 3
    num_phases = 2

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("real critic inputs must not be VAE reconstructions")

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(latent[:, :2], dim=1)


class GANTrainerTest(unittest.TestCase):
    def test_trains_and_saves_generator_and_critic(self):
        generator = LatentGenerator(
            latent_ch=3,
            latent_size=16,
            noise_ch=8,
            base_ch=8,
        )
        critic = ImageCritic(2, image_size=16, base_ch=4)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        images = torch.zeros(2, 1, 16, 16)
        images[:, :, 8:] = 1.0
        fake_volumes = torch.randn(2, 3, 16, 16, 16)
        generator_latents = []
        lmpdd_fakes = []
        penalty_fakes = []
        fake_score_batch_sizes = []
        hook = generator.register_forward_hook(
            lambda _module, _inputs, output: generator_latents.append(
                output.detach().clone()
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            trainer = GANTrainer(
                generator,
                critic,
                TinyVAE(),
                [images],
                list(fake_volumes),
                generator_optimizer,
                critic_optimizer,
                steps=1,
                critic_steps=1,
                gp_weight=10.0,
                clip_grad_norm=1.0,
                save_every=1,
                device="cpu",
                run_root=tmp,
            )
            next_lmpdd = trainer._next_fake_consensus_slices

            def track_lmpdd(*, count: int) -> torch.Tensor:
                fake = next_lmpdd(count=count)
                lmpdd_fakes.append(fake.detach().clone())
                return fake

            def track_penalty(
                critic_model: torch.nn.Module,
                real: torch.Tensor,
                fake: torch.Tensor,
            ) -> torch.Tensor:
                penalty_fakes.append(fake.detach().clone())
                return gradient_penalty(critic_model, real, fake)

            def track_critic_loss(
                real_scores: torch.Tensor,
                fake_scores: torch.Tensor,
                penalty: torch.Tensor,
                *,
                gp_weight: float,
            ) -> torch.Tensor:
                fake_score_batch_sizes.append(int(fake_scores.shape[0]))
                return critic_loss(
                    real_scores,
                    fake_scores,
                    penalty,
                    gp_weight=gp_weight,
                )

            trainer._next_fake_consensus_slices = track_lmpdd
            with (
                patch(
                    "src.pipeline.train.gan.gradient_penalty",
                    side_effect=track_penalty,
                ),
                patch(
                    "src.pipeline.train.gan.critic_loss",
                    side_effect=track_critic_loss,
                ),
            ):
                stats = trainer.train_step()
            checkpoint = torch.load(
                Path(trainer.run_dir) / "weight" / "gan" / "last" / "model.pt",
                map_location="cpu",
            )
            cached_index, cached_labels = next(iter(trainer.fake_volume_cache.items()))
            expected_labels = (
                decode_volume_probs(
                    TinyVAE(), fake_volumes[cached_index], num_phases=2
                )
                .argmax(dim=1)[0]
                .to(torch.uint8)
            )
            trainer.close()
        hook.remove()

        generator_probabilities = TinyVAE().decode_probs(generator_latents[0])
        expected_generator_fake = F.one_hot(
            generator_probabilities.argmax(dim=1),
            num_classes=2,
        ).movedim(-1, 1).to(generator_probabilities.dtype)

        self.assertEqual(trainer.step, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))
        self.assertIn("generator", checkpoint)
        self.assertIn("critic", checkpoint)
        self.assertNotIn("fraction_error", stats)
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(len(trainer.fake_volume_cache), 1)
        self.assertTrue(torch.equal(cached_labels, expected_labels))
        self.assertEqual(fake_score_batch_sizes, [2 * images.shape[0]])
        self.assertEqual(len(penalty_fakes), 2)
        self.assertTrue(torch.equal(penalty_fakes[0], expected_generator_fake))
        self.assertTrue(torch.equal(penalty_fakes[1], lmpdd_fakes[0]))


if __name__ == "__main__":
    unittest.main()
