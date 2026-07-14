import tempfile
import unittest
from pathlib import Path

import torch

from src.modeling.critic import LatentCritic, LatentGenerator
from src.pipelines.training import GANTrainer


class TinyVAE(torch.nn.Module):
    num_phases = 2

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.cat((images, -images), dim=1)
        return latent, torch.zeros_like(latent)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(latent, dim=1)


class GANTrainerTest(unittest.TestCase):
    def test_trains_and_saves_generator_and_critic(self):
        generator = LatentGenerator(
            latent_ch=2,
            latent_size=16,
            num_phases=2,
            noise_ch=8,
            base_ch=8,
        )
        critic = LatentCritic(2, 2, base_ch=4)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        images = torch.zeros(2, 1, 16, 16)
        images[:, :, 8:] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            trainer = GANTrainer(
                generator,
                critic,
                TinyVAE(),
                [images],
                generator_optimizer,
                critic_optimizer,
                steps=1,
                critic_steps=1,
                gradient_weight=10.0,
                fraction_weight=5.0,
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
            trainer.close()

        self.assertEqual(trainer.step, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))
        self.assertIn("generator", checkpoint)
        self.assertIn("critic", checkpoint)
        self.assertEqual(checkpoint["step"], 1)


if __name__ == "__main__":
    unittest.main()
