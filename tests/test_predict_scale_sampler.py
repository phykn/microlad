import unittest
from unittest.mock import patch

import torch

from src.predict.scale.sampler import sample_large_lmpdd


class IdentityDDPM:
    num_timesteps = 1

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start


class ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class ScaleSamplerTest(unittest.TestCase):
    def test_sample_large_lmpdd_injects_anchor_latent(self):
        anchor = torch.zeros(1, 4, 4, 4)
        anchor[:, 2, 1:3, 1:3] = 1
        mask = torch.zeros_like(anchor)
        mask[:, 2, 1:3, 1:3] = 1

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            latent = sample_large_lmpdd(
                ZeroModel(),
                IdentityDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=anchor,
                anchor_mask=mask,
            )

        self.assertTrue(torch.equal(latent[:, 2, 1:3, 1:3], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(latent[:, 0], torch.zeros(1, 4, 4)))

    def test_sample_large_lmpdd_rejects_partial_anchor_inputs(self):
        with self.assertRaisesRegex(ValueError, "anchor_latent and anchor_mask"):
            sample_large_lmpdd(
                ZeroModel(),
                IdentityDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=torch.zeros(1, 4, 4, 4),
            )


if __name__ == "__main__":
    unittest.main()
