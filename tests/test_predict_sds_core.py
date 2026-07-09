import unittest

import torch

from src.diffusion import DDPMProcess
from src.guidance.prior import sds_loss


class ConstantNoiseModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.grad_enabled: list[bool] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        return torch.ones_like(x) * self.weight


class PredictSDSCoreTest(unittest.TestCase):
    def test_sds_loss_uses_sds_gradient_direction(self):
        ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
        model = ConstantNoiseModel(value=0.25)
        latent = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
        noise = torch.full_like(latent, 1.25)
        t = torch.tensor([2], dtype=torch.long)

        loss, stats = sds_loss(
            latent,
            model,
            ddpm,
            t=t,
            noise=noise,
            t_min=1,
            t_max=3,
        )

        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
        expected = (sigma.pow(2) * latent * (0.25 - noise)).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(stats["sds"], expected.detach()))

    def test_sds_loss_gradient_sign_matches_sds_direction(self):
        ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
        model = ConstantNoiseModel(value=0.25)
        latent = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
        noise = torch.full_like(latent, 1.25)
        t = torch.tensor([2], dtype=torch.long)

        loss, _ = sds_loss(
            latent,
            model,
            ddpm,
            t=t,
            noise=noise,
            t_min=1,
            t_max=3,
        )
        loss.backward()

        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
        expected_grad = sigma.pow(2) * (0.25 - noise) / latent.numel()
        self.assertTrue(torch.allclose(latent.grad, expected_grad))
        self.assertTrue(torch.all(latent.detach() - latent.grad > latent.detach()))

    def test_sds_loss_spatial_weight_limits_gradient_support(self):
        ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
        model = ConstantNoiseModel(value=0.25)
        latent = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
        noise = torch.full_like(latent, 1.25)
        timestep = torch.tensor([2], dtype=torch.long)
        spatial_weight = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

        loss, _ = sds_loss(
            latent,
            model,
            ddpm,
            t=timestep,
            noise=noise,
            t_min=1,
            t_max=3,
            spatial_weight=spatial_weight,
            spatial_normalizer=1.0,
        )
        loss.backward()

        self.assertNotEqual(float(latent.grad[0, 0, 0, 0]), 0.0)
        self.assertTrue(torch.equal(latent.grad[0, 0, 0, 1:], torch.zeros(1)))
        self.assertTrue(torch.equal(latent.grad[0, 0, 1], torch.zeros(2)))

    def test_sds_loss_does_not_backpropagate_through_model(self):
        ddpm = DDPMProcess(timesteps=4)
        model = ConstantNoiseModel(value=0.0)
        latent = torch.randn(1, 1, 2, 2, requires_grad=True)

        loss, _ = sds_loss(latent, model, ddpm, t_min=1, t_max=3)
        loss.backward()

        self.assertIsNotNone(latent.grad)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(model.grad_enabled, [False])

    def test_sds_loss_samples_timesteps_inside_requested_range(self):
        ddpm = DDPMProcess(timesteps=8)
        model = ConstantNoiseModel(value=0.0)
        latent = torch.randn(4, 1, 2, 2, requires_grad=True)

        _, stats = sds_loss(latent, model, ddpm, t_min=2, t_max=5)

        self.assertGreaterEqual(int(stats["t"].min()), 2)
        self.assertLess(int(stats["t"].max()), 5)

    def test_sds_loss_rejects_invalid_inputs(self):
        ddpm = DDPMProcess(timesteps=4)
        model = ConstantNoiseModel(value=0.0)

        with self.assertRaisesRegex(ValueError, "latent"):
            sds_loss(torch.zeros(1, 2, 2), model, ddpm, t_min=1, t_max=3)
        with self.assertRaisesRegex(ValueError, "positive"):
            sds_loss(torch.empty(0, 1, 2, 2), model, ddpm, t_min=1, t_max=3)
        with self.assertRaisesRegex(ValueError, "positive"):
            sds_loss(torch.empty(1, 1, 0, 2), model, ddpm, t_min=1, t_max=3)
        with self.assertRaisesRegex(ValueError, "timestep"):
            sds_loss(torch.zeros(1, 1, 2, 2), model, ddpm, t_min=3, t_max=3)
        with self.assertRaisesRegex(ValueError, "noise"):
            sds_loss(
                torch.zeros(1, 1, 2, 2),
                model,
                ddpm,
                t_min=1,
                t_max=3,
                noise=torch.zeros(1, 1, 1, 1),
            )

    def test_sds_loss_rejects_non_finite_values(self):
        ddpm = DDPMProcess(timesteps=4)
        valid_latent = torch.zeros(1, 1, 2, 2)
        valid_noise = torch.zeros_like(valid_latent)
        valid_t = torch.tensor([1], dtype=torch.long)

        cases = [
            lambda: sds_loss(
                torch.full_like(valid_latent, float("nan")),
                ConstantNoiseModel(value=0.0),
                ddpm,
                t=valid_t,
                noise=valid_noise,
                t_min=1,
                t_max=3,
            ),
            lambda: sds_loss(
                valid_latent,
                ConstantNoiseModel(value=0.0),
                ddpm,
                t=valid_t,
                noise=torch.full_like(valid_latent, float("inf")),
                t_min=1,
                t_max=3,
            ),
            lambda: sds_loss(
                valid_latent,
                ConstantNoiseModel(value=float("inf")),
                ddpm,
                t=valid_t,
                noise=valid_noise,
                t_min=1,
                t_max=3,
            ),
        ]

        for call in cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "finite"):
                    call()


if __name__ == "__main__":
    unittest.main()
