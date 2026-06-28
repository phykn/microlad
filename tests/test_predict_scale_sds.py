import unittest

import numpy as np
import torch

from src.models import DDPM
from src.predict import AnchorSlice
from src.predict.scale import optimize_large_volume
from src.predict.scale.sds import _local_prior_objective


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1

    def encode(self, image: torch.Tensor):
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class PredictScaleSDSTest(unittest.TestCase):
    def test_optimize_large_volume_updates_scheduled_anchor_slice_tiles(self):
        volume = torch.zeros(4, 4, 4)
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=2,
        )

        updated, stats = optimize_large_volume(
            volume,
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            steps=1,
            slice_steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 2)],
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
            tile_overlap=0,
        )

        self.assertGreater(float(updated[2].mean()), 0.0)
        self.assertLess(float(updated[2].mean()), 1.0)
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertIn("anchor", stats)
        self.assertIn("steps", stats)

    def test_optimize_large_volume_applies_full_slice_vf_target_loss(self):
        updated, stats = optimize_large_volume(
            torch.zeros(4, 4, 4),
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            steps=1,
            slice_steps=2,
            lr=0.5,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 1)],
            sds_weight=0.0,
            vf_targets=torch.tensor([1.0, 0.0]),
            vf_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )

        self.assertLess(float(updated[1].mean()), 0.0)
        self.assertIn("vf", stats)
        self.assertIn("loss", stats)

    def test_large_slice_prior_loss_is_averaged_across_tiles(self):
        decoded, total, stats = _local_prior_objective(
            torch.zeros(4, 4),
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=torch.ones(4, 4),
            anchor_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )

        self.assertEqual(decoded.shape, torch.Size([4, 4]))
        self.assertIn("anchor", stats)
        self.assertTrue(torch.allclose(total.detach(), stats["anchor"]))


if __name__ == "__main__":
    unittest.main()
