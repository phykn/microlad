import unittest

import torch

from src.models import PatchVAE, reparameterize


class PatchVAETest(unittest.TestCase):
    def test_vae_rejects_invalid_latent_channels(self):
        with self.assertRaisesRegex(ValueError, "latent_ch"):
            PatchVAE(latent_ch=0)

    def test_encode_decode_shapes_match_microstructure_contract(self):
        vae = PatchVAE(latent_ch=4)
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
        vae = PatchVAE(latent_ch=4)
        x = torch.randn(2, 1, 64, 64)

        recon, mu, logvar = vae(x)

        self.assertEqual(recon.shape, torch.Size([2, 1, 64, 64]))
        self.assertEqual(mu.shape, torch.Size([2, 4, 16, 16]))
        self.assertEqual(logvar.shape, torch.Size([2, 4, 16, 16]))

    def test_encode_requires_gray_image_batch(self):
        vae = PatchVAE(latent_ch=4)

        with self.assertRaisesRegex(ValueError, "gray image batch"):
            vae.encode(torch.randn(1, 64, 64))
        with self.assertRaisesRegex(ValueError, "gray image batch"):
            vae.encode(torch.randn(1, 3, 64, 64))

    def test_encode_requires_size_divisible_by_four(self):
        vae = PatchVAE(latent_ch=4)

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            vae.encode(torch.randn(1, 1, 63, 64))
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            vae.encode(torch.randn(1, 1, 64, 63))

    def test_decode_requires_matching_latent_batch(self):
        vae = PatchVAE(latent_ch=4)

        with self.assertRaisesRegex(ValueError, "latent batch"):
            vae.decode(torch.randn(4, 16, 16))
        with self.assertRaisesRegex(ValueError, "latent batch"):
            vae.decode(torch.randn(1, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
