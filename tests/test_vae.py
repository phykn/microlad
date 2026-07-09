import unittest

import torch
import torch.nn as nn

from src.neural import downsample_steps
from src.vae import PatchVAE, reparameterize


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
    def test_bottleneck_attention_runs_in_encoder_and_decoder(self):
        model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=32)
        attention_blocks = [
            module
            for module in model.modules()
            if type(module).__name__ == "AttentionBlock"
        ]
        calls = []

        self.assertEqual(len(attention_blocks), 2)

        for module in attention_blocks:
            module.register_forward_hook(
                lambda _module, _inputs, _output: calls.append(_module)
            )

        model(torch.randn(1, 1, 64, 64))

        self.assertEqual(calls, attention_blocks)

    def test_down_and_up_blocks_run_two_residual_blocks_before_resampling(self):
        model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=32)

        for block in [*model.down_blocks, *model.up_blocks]:
            residual_blocks = [
                module
                for module in block.modules()
                if type(module).__name__ == "ResidualBlock"
            ]
            self.assertEqual(len(residual_blocks), 2)

        first_down_shapes = []
        first_up_shapes = []
        for module in model.down_blocks[0].modules():
            if type(module).__name__ == "ResidualBlock":
                module.register_forward_hook(
                    lambda _module, inputs, _output: first_down_shapes.append(
                        tuple(inputs[0].shape[-2:])
                    )
                )
        for module in model.up_blocks[0].modules():
            if type(module).__name__ == "ResidualBlock":
                module.register_forward_hook(
                    lambda _module, inputs, _output: first_up_shapes.append(
                        tuple(inputs[0].shape[-2:])
                    )
                )

        model(torch.randn(1, 1, 64, 64))

        self.assertEqual(first_down_shapes, [(64, 64), (64, 64)])
        self.assertEqual(first_up_shapes, [(16, 16), (16, 16)])

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
                    num_phases=3,
                    base_ch=8,
                    max_ch=32,
                )
                x = torch.randn(1, 1, image_size, image_size)

                logits, mu, logvar = model(x)

                self.assertEqual(logits.shape, torch.Size([1, 3, image_size, image_size]))
                self.assertEqual(mu.shape, torch.Size([1, 4, 16, 16]))
                self.assertEqual(logvar.shape, torch.Size([1, 4, 16, 16]))
                self.assertEqual(model.downsample_factor, image_size // 16)
                self.assertEqual(model.downsample_steps, downsample_steps(image_size, 16))

    def test_decode_logits_returns_phase_channels_and_decode_returns_expected_phase_image(self):
        model = PatchVAE(
            image_size=64,
            latent_size=16,
            latent_ch=4,
            num_phases=4,
            base_ch=8,
            max_ch=32,
        )
        latent = torch.randn(2, 4, 16, 16)

        logits = model.decode_logits(latent)
        decoded = model.decode(latent)

        self.assertEqual(logits.shape, torch.Size([2, 4, 64, 64]))
        self.assertEqual(decoded.shape, torch.Size([2, 1, 64, 64]))
        self.assertTrue(torch.all(decoded >= 0.0))
        self.assertTrue(torch.all(decoded <= 3.0))

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
