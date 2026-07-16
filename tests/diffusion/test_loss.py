import unittest

import torch

from src.diffusion import DDPMProcess, DiffusionLoss, compute_loss


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


class AxisRecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_fractions = None
        self.seen_axis_condition = None

    def forward(self, x, t, fractions, axis_condition):
        self.seen_fractions = fractions
        self.seen_axis_condition = axis_condition
        return torch.zeros_like(x)


class DiffusionLossTest(unittest.TestCase):
    def test_compute_loss_is_zero_when_model_predicts_exact_noise(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 3], dtype=torch.long)
        model = FixedNoiseModel(noise)

        loss, parts = compute_loss(model, ddpm, clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(0.0)))
        self.assertIs(model.seen_t, t)

    def test_compute_loss_matches_mse_between_predicted_and_true_noise(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.ones_like(clean)
        t = torch.tensor([0, 2], dtype=torch.long)

        loss, parts = compute_loss(ZeroNoiseModel(), ddpm, clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(1.0)))

    def test_compute_loss_samples_timestep_and_noise_when_not_given(self):
        ddpm = DDPMProcess(timesteps=4)
        model = ZeroNoiseModel()
        clean = torch.randn(2, 4, 8, 8)

        loss, parts = compute_loss(model, ddpm, clean)

        self.assertEqual(loss.ndim, 0)
        self.assertIn("noise", parts)
        self.assertGreaterEqual(float(loss.detach()), 0.0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for device mismatch"
    )
    def test_compute_loss_rejects_timestep_on_wrong_device(self):
        ddpm = DDPMProcess(timesteps=4, device="cuda")
        model = ZeroNoiseModel().cuda()
        clean = torch.randn(2, 4, 8, 8, device="cuda")
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long, device="cpu")

        with self.assertRaisesRegex(ValueError, "device"):
            compute_loss(model, ddpm, clean, t=t, noise=noise)

    def test_loss_module_wraps_function(self):
        ddpm = DDPMProcess(timesteps=4)
        loss_fn = DiffusionLoss(ddpm)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.ones_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long)

        loss, parts = loss_fn(ZeroNoiseModel(), clean, t=t, noise=noise)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(1.0)))

    def test_compute_loss_rejects_invalid_inputs(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.randn(2, 4, 8, 8)
        noise = torch.randn_like(clean)
        t = torch.tensor([1, 2], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "image"):
            compute_loss(ZeroNoiseModel(), ddpm, torch.randn(2, 4, 8), t=t)
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_loss(
                ZeroNoiseModel(),
                ddpm,
                torch.empty(0, 4, 8, 8),
                t=torch.empty(0, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "noise"):
            compute_loss(ZeroNoiseModel(), ddpm, clean, t=t, noise=noise[:, :1])
        with self.assertRaisesRegex(ValueError, "model output"):
            compute_loss(BadShapeModel(), ddpm, clean, t=t, noise=noise)

    def test_compute_loss_forwards_axis_and_reports_per_axis_losses(self):
        ddpm = DDPMProcess(timesteps=4)
        model = AxisRecordingModel()
        clean = torch.zeros(3, 2, 4, 4)
        noise = torch.stack(
            [
                torch.zeros(2, 4, 4),
                torch.ones(2, 4, 4),
                torch.full((2, 4, 4), 2.0),
            ]
        )
        fractions = torch.tensor([[0.5, 0.5]]).expand(3, -1)
        axis_condition = torch.tensor([0, 1, 2])

        loss, parts = compute_loss(
            model,
            ddpm,
            clean,
            fractions=fractions,
            t=torch.tensor([0, 1, 2]),
            noise=noise,
            axis_condition=axis_condition,
        )

        self.assertIs(model.seen_fractions, fractions)
        self.assertIs(model.seen_axis_condition, axis_condition)
        self.assertTrue(torch.allclose(loss, torch.tensor(5.0 / 3.0)))
        self.assertTrue(torch.allclose(parts["axis_0"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(parts["axis_1"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["axis_2"], torch.tensor(4.0)))


if __name__ == "__main__":
    unittest.main()
