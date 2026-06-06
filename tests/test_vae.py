import unittest

import torch

from models import CustomVAE, reparameterize


class CustomVAETest(unittest.TestCase):
    def test_encode_decode_shapes_match_microstructure_contract(self):
        vae = CustomVAE(latent_ch=4)
        x = torch.randn(2, 1, 64, 64)

        mu, logvar = vae.encode(x)
        z = reparameterize(mu, logvar)
        recon = vae.decode(z)

        self.assertEqual(mu.shape, torch.Size([2, 4, 16, 16]))
        self.assertEqual(logvar.shape, torch.Size([2, 4, 16, 16]))
        self.assertEqual(z.shape, torch.Size([2, 4, 16, 16]))
        self.assertEqual(recon.shape, torch.Size([2, 1, 64, 64]))
        self.assertGreaterEqual(float(recon.detach().min()), 0.0)
        self.assertLessEqual(float(recon.detach().max()), 1.0)

    def test_forward_returns_reconstruction_and_latent_distribution(self):
        vae = CustomVAE(latent_ch=4)
        x = torch.randn(2, 1, 64, 64)

        recon, mu, logvar = vae(x)

        self.assertEqual(recon.shape, torch.Size([2, 1, 64, 64]))
        self.assertEqual(mu.shape, torch.Size([2, 4, 16, 16]))
        self.assertEqual(logvar.shape, torch.Size([2, 4, 16, 16]))


if __name__ == "__main__":
    unittest.main()
