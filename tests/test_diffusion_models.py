import unittest

import torch

from src.models import DDPM, TimeUNet


class DDPMTest(unittest.TestCase):
    def test_q_sample_uses_closed_form_noise_formula(self):
        ddpm = DDPM(timesteps=4, beta_start=0.1, beta_end=0.2)
        x_start = torch.ones(2, 3, 4, 4)
        noise = torch.full_like(x_start, 2.0)
        t = torch.tensor([0, 3], dtype=torch.long)

        noisy = ddpm.q_sample(x_start, t, noise=noise)

        expected = (
            ddpm.sqrt_alphas_cumprod[t].view(2, 1, 1, 1) * x_start
            + ddpm.sqrt_one_minus_alphas_cumprod[t].view(2, 1, 1, 1) * noise
        )
        self.assertTrue(torch.allclose(noisy, expected))

    def test_add_noise_matches_q_sample(self):
        ddpm = DDPM(timesteps=4)
        x_start = torch.randn(2, 3, 4, 4)
        noise = torch.randn_like(x_start)
        t = torch.tensor([1, 2], dtype=torch.long)

        self.assertTrue(
            torch.allclose(
                ddpm.add_noise(x_start, t, noise=noise),
                ddpm.q_sample(x_start, t, noise=noise),
            )
        )

    def test_sample_timesteps_returns_integer_batch(self):
        ddpm = DDPM(timesteps=8)

        t = ddpm.sample_timesteps(batch_size=5)

        self.assertEqual(t.shape, torch.Size([5]))
        self.assertEqual(t.dtype, torch.long)
        self.assertGreaterEqual(int(t.min()), 0)
        self.assertLess(int(t.max()), 8)

    def test_p_sample_preserves_shape(self):
        class ZeroNoise(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        ddpm = DDPM(timesteps=4)
        x = torch.randn(2, 3, 4, 4)
        t = torch.tensor([0, 2], dtype=torch.long)

        sample = ddpm.p_sample(ZeroNoise(), x, t)

        self.assertEqual(sample.shape, x.shape)

    def test_p_mean_matches_ddpm_posterior_mean_formula(self):
        class FixedNoise(torch.nn.Module):
            def forward(self, x, t):
                return torch.full_like(x, 0.25)

        ddpm = DDPM(timesteps=4, beta_start=0.1, beta_end=0.2)
        x = torch.full((2, 3, 4, 4), 0.5)
        t = torch.tensor([0, 2], dtype=torch.long)

        mean = ddpm.p_mean(FixedNoise(), x, t)

        alpha = ddpm.alphas[t].view(2, 1, 1, 1)
        beta = ddpm.betas[t].view(2, 1, 1, 1)
        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(2, 1, 1, 1)
        expected = (x - beta / sigma * 0.25) / torch.sqrt(alpha)
        self.assertTrue(torch.allclose(mean, expected))

    def test_rejects_invalid_timesteps(self):
        ddpm = DDPM(timesteps=4)
        x = torch.randn(2, 3, 4, 4)

        with self.assertRaisesRegex(ValueError, "shape"):
            ddpm.q_sample(x, torch.tensor([[0, 1]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "integer"):
            ddpm.q_sample(x, torch.tensor([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "schedule"):
            ddpm.q_sample(x, torch.tensor([0, 4], dtype=torch.long))

    def test_rejects_empty_batch_timesteps(self):
        ddpm = DDPM(timesteps=4)
        empty = torch.empty(0, 3, 4, 4)
        t = torch.empty(0, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "batch"):
            ddpm.q_sample(empty, t)
        with self.assertRaisesRegex(ValueError, "batch"):
            ddpm.p_sample(torch.nn.Identity(), empty, t)


class TimeUNetTest(unittest.TestCase):
    def test_attention_runs_at_each_latent_scale(self):
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)
        attention_blocks = [
            module
            for module in model.modules()
            if type(module).__name__ == "SelfAttention"
        ]
        calls = []

        self.assertEqual(len(attention_blocks), 3)

        for module in attention_blocks:
            module.register_forward_hook(
                lambda _module, _inputs, _output: calls.append(_module)
            )

        model(torch.randn(1, 4, 16, 16), torch.tensor([0], dtype=torch.long))

        self.assertEqual(calls, attention_blocks)

    def test_each_unet_level_uses_two_time_residual_blocks(self):
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)

        for name in ("enc1", "enc2", "mid", "dec2", "dec1"):
            with self.subTest(level=name):
                residual_blocks = [
                    module
                    for module in getattr(model, name).modules()
                    if type(module).__name__ == "TimeResidualBlock"
                ]

                self.assertEqual(len(residual_blocks), 2)

    def test_forward_predicts_noise_with_same_shape_as_latent(self):
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)
        x = torch.randn(2, 4, 16, 16)
        t = torch.tensor([0, 9], dtype=torch.long)

        noise = model(x, t)

        self.assertEqual(noise.shape, x.shape)

    def test_rejects_wrong_latent_shape(self):
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "positive"):
            model(torch.empty(1, 4, 0, 16), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(torch.randn(2, 4, 15, 16), torch.tensor([0, 1]))

    def test_rejects_wrong_timestep_shape(self):
        model = TimeUNet(latent_ch=4, base_ch=8, time_dim=16)

        with self.assertRaisesRegex(ValueError, "timesteps"):
            model(torch.randn(2, 4, 16, 16), torch.tensor([[0, 1]]))


if __name__ == "__main__":
    unittest.main()
