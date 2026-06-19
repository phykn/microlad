import unittest

import torch

from src.models import DDPM, TimeUNet


class TimeUNetTest(unittest.TestCase):
    def test_rejects_invalid_model_dimensions(self):
        with self.assertRaisesRegex(ValueError, "latent_ch"):
            TimeUNet(latent_ch=0, base_ch=16, time_dim=16)
        with self.assertRaisesRegex(ValueError, "base_ch"):
            TimeUNet(latent_ch=4, base_ch=0, time_dim=16)
        with self.assertRaisesRegex(ValueError, "time_dim"):
            TimeUNet(latent_ch=4, base_ch=16, time_dim=0)

    def test_predicts_noise_with_same_latent_shape(self):
        unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
        z_t = torch.randn(2, 4, 16, 16)
        t = torch.tensor([0, 9])

        pred = unet(z_t, t)

        self.assertEqual(pred.shape, z_t.shape)

    def test_accepts_non_default_channel_and_time_dimensions(self):
        unet = TimeUNet(latent_ch=4, base_ch=24, time_dim=15)
        z_t = torch.randn(2, 4, 16, 16)
        t = torch.tensor([0, 9])

        pred = unet(z_t, t)

        self.assertEqual(pred.shape, z_t.shape)

    def test_forward_requires_matching_latent_batch_and_timesteps(self):
        unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)

        with self.assertRaisesRegex(ValueError, "latent batch"):
            unet(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "timesteps"):
            unet(torch.randn(2, 4, 16, 16), torch.tensor([0]))

    def test_forward_requires_size_divisible_by_four(self):
        unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            unet(torch.randn(2, 4, 15, 16), torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            unet(torch.randn(2, 4, 16, 15), torch.tensor([0, 1]))


class DDPMTest(unittest.TestCase):
    def test_rejects_invalid_schedule_values(self):
        with self.assertRaisesRegex(ValueError, "timesteps"):
            DDPM(timesteps=0)
        with self.assertRaisesRegex(ValueError, "beta_start"):
            DDPM(timesteps=10, beta_start=0.0)
        with self.assertRaisesRegex(ValueError, "beta_end"):
            DDPM(timesteps=10, beta_end=1.0)
        with self.assertRaisesRegex(ValueError, "beta_start"):
            DDPM(timesteps=10, beta_start=0.02, beta_end=0.01)

    def test_q_sample_and_p_sample_keep_latent_shape(self):
        ddpm = DDPM(timesteps=10)
        unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16).eval()
        z = torch.randn(2, 4, 16, 16)
        t = torch.tensor([0, 9])
        noise = torch.randn_like(z)

        z_t = ddpm.q_sample(z, t, noise)
        with torch.no_grad():
            z_prev = ddpm.p_sample(unet, z_t, t)

        self.assertEqual(z_t.shape, z.shape)
        self.assertEqual(z_prev.shape, z.shape)

    def test_q_sample_validates_timesteps_and_noise_shape(self):
        ddpm = DDPM(timesteps=10)
        z = torch.randn(2, 4, 16, 16)

        with self.assertRaisesRegex(ValueError, "timesteps"):
            ddpm.q_sample(z, torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "integer"):
            ddpm.q_sample(z, torch.tensor([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "timestep values"):
            ddpm.q_sample(z, torch.tensor([0, 10]))
        with self.assertRaisesRegex(ValueError, "noise"):
            ddpm.q_sample(z, torch.tensor([0, 1]), torch.randn(1, 4, 16, 16))

    def test_p_sample_validates_timesteps(self):
        class ZeroNoiseModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        ddpm = DDPM(timesteps=10)
        z_t = torch.randn(2, 4, 16, 16)

        with self.assertRaisesRegex(ValueError, "timesteps"):
            ddpm.p_sample(ZeroNoiseModel(), z_t, torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "timestep values"):
            ddpm.p_sample(ZeroNoiseModel(), z_t, torch.tensor([0, 10]))

    def test_q_sample_uses_input_device(self):
        ddpm = DDPM(timesteps=10)
        z = torch.empty(2, 4, 16, 16, device="meta")
        t = torch.tensor([0, 9])
        noise = torch.empty_like(z)

        z_t = ddpm.q_sample(z, t, noise)

        self.assertEqual(z_t.device, z.device)

    def test_p_sample_uses_input_device(self):
        class ZeroNoiseModel(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)

        ddpm = DDPM(timesteps=10)
        z_t = torch.empty(2, 4, 16, 16, device="meta")
        t = torch.tensor([0, 9])

        z_prev = ddpm.p_sample(ZeroNoiseModel(), z_t, t)

        self.assertEqual(z_prev.device, z_t.device)


if __name__ == "__main__":
    unittest.main()
