import unittest
from unittest.mock import patch

import torch

from src.models import DDPM
from src.predict.sampler import DiffusionSampler


class RecordingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.steps.append(int(t[0].item()))
        return torch.zeros_like(x)


class GradCheckDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_enabled: list[bool] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        return x * 0.0 * self.weight


class IdentityDDPM:
    def __init__(self, timesteps: int) -> None:
        self.num_timesteps = timesteps
        self.steps: list[int] = []

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.steps.append(int(t[0].item()))
        return x


class PredictSamplerTest(unittest.TestCase):
    def test_sample_returns_requested_latent_shape(self):
        model = RecordingDenoiser()
        sampler = DiffusionSampler(model, DDPM(timesteps=4), device="cpu")

        latent = sampler.sample((2, 3, 4, 4))

        self.assertEqual(latent.shape, torch.Size([2, 3, 4, 4]))
        self.assertEqual(model.steps, [3, 2, 1, 0])

    def test_sample_runs_without_gradients(self):
        model = GradCheckDenoiser()
        sampler = DiffusionSampler(model, DDPM(timesteps=2), device="cpu")

        latent = sampler.sample((1, 1, 4, 4))

        self.assertFalse(latent.requires_grad)
        self.assertEqual(model.grad_enabled, [False, False])

    def test_sample_rejects_invalid_shape(self):
        sampler = DiffusionSampler(RecordingDenoiser(), DDPM(timesteps=2), device="cpu")

        with self.assertRaisesRegex(ValueError, "shape"):
            sampler.sample((1, 4, 4))

    def test_sample_lmpdd_returns_canonical_axis_order_after_rotating_between_steps(self):
        base = torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2)
        ddpm = IdentityDDPM(timesteps=3)
        sampler = DiffusionSampler(RecordingDenoiser(), ddpm, device="cpu")

        with patch("torch.randn", return_value=base.clone()):
            latent = sampler.sample_lmpdd((2, 1, 2, 2))

        self.assertTrue(torch.equal(latent, base))
        self.assertEqual(ddpm.steps, [2, 1, 0])

    def test_sample_lmpdd_blends_anchor_latent(self):
        sampler = DiffusionSampler(
            RecordingDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )
        anchor_latent = torch.zeros(2, 1, 2, 2)
        anchor_latent[1] = 1.0
        anchor_mask = torch.zeros(2, 1, 2, 2)
        anchor_mask[1] = 1.0

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            latent = sampler.sample_lmpdd(
                (2, 1, 2, 2),
                anchor_latent=anchor_latent,
                anchor_mask=anchor_mask,
            )

        self.assertTrue(torch.equal(latent[1], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(latent[0], torch.zeros(1, 2, 2)))

    def test_sample_lmpdd_rejects_non_cubic_latent_shape(self):
        sampler = DiffusionSampler(RecordingDenoiser(), DDPM(timesteps=2), device="cpu")

        with self.assertRaisesRegex(ValueError, "cubic"):
            sampler.sample_lmpdd((2, 1, 3, 2))


if __name__ == "__main__":
    unittest.main()
