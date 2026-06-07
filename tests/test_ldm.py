import unittest

import torch

from src.models import DDPM, TimeUNet


class TimeUNetTest(unittest.TestCase):
    def test_predicts_noise_with_same_latent_shape(self):
        unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
        z_t = torch.randn(2, 4, 16, 16)
        t = torch.tensor([0, 9])

        pred = unet(z_t, t)

        self.assertEqual(pred.shape, z_t.shape)

class DDPMTest(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
