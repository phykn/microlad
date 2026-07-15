import tempfile
import unittest
from pathlib import Path

import torch

from src.modeling.latent_gan import ImageCritic, LatentGenerator
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

        self.assertEqual(trainer.step, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))
        self.assertIn("generator", checkpoint)
        self.assertIn("critic", checkpoint)
        self.assertNotIn("fraction_error", stats)
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(len(trainer.fake_volume_cache), 1)
        self.assertTrue(torch.equal(cached_labels, expected_labels))


if __name__ == "__main__":
    unittest.main()
