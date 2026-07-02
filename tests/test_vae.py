import unittest

import torch
import torch.nn as nn

from src.models import PatchVAE, downsample_steps, reparameterize


class DownsampleShapeTest(unittest.TestCase):
    def test_downsample_steps_returns_power_of_two_steps(self):
        self.assertEqual(downsample_steps(image_size=64, latent_size=16), 2)
        self.assertEqual(downsample_steps(image_size=128, latent_size=16), 3)
        self.assertEqual(downsample_steps(image_size=256, latent_size=16), 4)

    def test_downsample_steps_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "greater"):
            downsample_steps(image_size=16, latent_size=16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            downsample_steps(image_size=130, latent_size=16)
        with self.assertRaisesRegex(ValueError, "power of two"):
            downsample_steps(image_size=96, latent_size=16)


class PatchVAETest(unittest.TestCase):
    def test_default_image_size_matches_original_repo_size(self):
        model = PatchVAE(base_ch=8, max_ch=32)

        self.assertEqual(model.image_size, 64)
        self.assertEqual(model.latent_size, 16)
        self.assertEqual(model.downsample_factor, 4)
        self.assertEqual(model.downsample_steps, 2)

    def test_rejects_non_positive_latent_size_with_value_error(self):
        with self.assertRaisesRegex(ValueError, "latent_size"):
            PatchVAE(image_size=64, latent_size=0, base_ch=8, max_ch=32)

    def test_forward_preserves_input_shape_for_supported_image_sizes(self):
        for image_size in (64, 128, 256):
            with self.subTest(image_size=image_size):
                model = PatchVAE(
                    image_size=image_size,
                    latent_size=16,
                    latent_ch=4,
                    base_ch=8,
                    max_ch=32,
                )
                x = torch.randn(1, 1, image_size, image_size)

                recon, mu, logvar = model(x)

                self.assertEqual(recon.shape, x.shape)
                self.assertEqual(mu.shape, torch.Size([1, 4, 16, 16]))
                self.assertEqual(logvar.shape, torch.Size([1, 4, 16, 16]))
                self.assertEqual(model.downsample_factor, image_size // 16)
                self.assertEqual(model.downsample_steps, downsample_steps(image_size, 16))

    def test_output_has_no_tanh_activation(self):
        model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=32)

        has_tanh = any(isinstance(module, nn.Tanh) for module in model.modules())

        self.assertFalse(has_tanh)

    def test_encode_rejects_wrong_input_shape(self):
        model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=32)

        with self.assertRaisesRegex(ValueError, "shape"):
            model.encode(torch.randn(1, 3, 64, 64))
        with self.assertRaisesRegex(ValueError, "64x64"):
            model.encode(torch.randn(1, 1, 32, 32))

    def test_decode_rejects_wrong_latent_shape(self):
        model = PatchVAE(
            image_size=64,
            latent_size=16,
            latent_ch=4,
            base_ch=8,
            max_ch=32,
        )

        with self.assertRaisesRegex(ValueError, "latent"):
            model.decode(torch.randn(1, 3, 16, 16))
        with self.assertRaisesRegex(ValueError, "16x16"):
            model.decode(torch.randn(1, 4, 8, 8))

    def test_reparameterize_preserves_shape(self):
        mu = torch.zeros(2, 4, 16, 16)
        logvar = torch.zeros_like(mu)

        z = reparameterize(mu, logvar)

        self.assertEqual(z.shape, mu.shape)

    def test_reparameterize_rejects_mismatched_shapes(self):
        mu = torch.zeros(2, 4, 16, 16)
        logvar = torch.zeros(1, 4, 16, 16)

        with self.assertRaisesRegex(ValueError, "shape"):
            reparameterize(mu, logvar)


if __name__ == "__main__":
    unittest.main()
