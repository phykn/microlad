import unittest

import torch

from src.loss import DiffusionLoss, diffusion_loss
from src.models import DDPM, TimeUNet


class FixedNoiseModel(torch.nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise
        self.seen_t = None

    def forward(self, x, t):
        self.seen_t = t
        return self.noise


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class BadShapeModel(torch.nn.Module):
    def forward(self, x, t):
        return x[:, :1]


class DiffusionLossTest(unittest.TestCase):
    def test_diffusion_loss_is_zero_when_model_predicts_exact_noise(self):
        ddpm = DDPM(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 3], dtype=torch.long)
        model = FixedNoiseModel(noise)

        loss, parts = diffusion_loss(model, ddpm, clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(0.0)))
        self.assertIs(model.seen_t, t)

    def test_diffusion_loss_matches_mse_between_predicted_and_true_noise(self):
        ddpm = DDPM(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.ones_like(clean)
        t = torch.tensor([0, 2], dtype=torch.long)

        loss, parts = diffusion_loss(ZeroNoiseModel(), ddpm, clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(1.0)))

    def test_diffusion_loss_samples_timestep_and_noise_when_not_given(self):
        ddpm = DDPM(timesteps=4)
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)
        clean = torch.randn(2, 4, 8, 8)

        loss, parts = diffusion_loss(model, ddpm, clean)

        self.assertEqual(loss.ndim, 0)
        self.assertIn("noise", parts)
        self.assertGreaterEqual(float(loss.detach()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device mismatch")
    def test_diffusion_loss_rejects_timestep_on_wrong_device(self):
        ddpm = DDPM(timesteps=4, device="cuda")
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16).cuda()
        clean = torch.randn(2, 4, 8, 8, device="cuda")
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long, device="cpu")

        with self.assertRaisesRegex(ValueError, "device"):
            diffusion_loss(model, ddpm, clean, t=t, noise=noise)

    def test_diffusion_loss_module_wraps_function(self):
        ddpm = DDPM(timesteps=4)
        loss_fn = DiffusionLoss(ddpm)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.ones_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long)

        loss, parts = loss_fn(ZeroNoiseModel(), clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(1.0)))

    def test_diffusion_loss_rejects_invalid_inputs(self):
        ddpm = DDPM(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "latent"):
            diffusion_loss(ZeroNoiseModel(), ddpm, torch.randn(2, 4, 8), t=t)
        with self.assertRaisesRegex(ValueError, "positive"):
            diffusion_loss(
                ZeroNoiseModel(),
                ddpm,
                torch.empty(0, 4, 8, 8),
                t=torch.empty(0, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "noise"):
            diffusion_loss(ZeroNoiseModel(), ddpm, clean, t=t, noise=noise[:, :1])
        with self.assertRaisesRegex(ValueError, "model output"):
            diffusion_loss(BadShapeModel(), ddpm, clean, t=t, noise=noise)


if __name__ == "__main__":
    unittest.main()
