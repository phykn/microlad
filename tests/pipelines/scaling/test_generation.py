import unittest

import torch
import torch.nn.functional as F

from src.modeling.slicegan import SliceGANGenerator
from src.pipelines.scaling.generation import render_generator_tiled


class GeneratorRenderingTest(unittest.TestCase):
    def test_rejects_base_generator_with_global_interpolation(self):
        with torch.device("meta"):
            generator = SliceGANGenerator(2).eval()
            noise = torch.empty(1, 32, 4, 4, 4, device="meta")

        with self.assertRaisesRegex(ValueError, "fully_convolutional"):
            render_generator_tiled(generator, noise)

    def test_tiled_render_matches_full_local_generator_without_seams(self):
        class LocalGenerator(torch.nn.Module):
            def forward(self, noise):
                smoothed = F.avg_pool3d(
                    F.pad(noise, (1, 1, 1, 1, 1, 1), mode="replicate"),
                    kernel_size=3,
                    stride=1,
                )
                return F.interpolate(
                    smoothed,
                    scale_factor=16,
                    mode="trilinear",
                    align_corners=False,
                )

        generator = LocalGenerator().eval()
        noise = torch.randn(1, 2, 8, 8, 8)

        full = generator(noise)
        tiled = render_generator_tiled(
            generator,
            noise,
            core_noise_size=4,
            halo_noise_size=2,
            output_device="cpu",
        )

        self.assertEqual(tiled.shape, torch.Size([1, 2, 128, 128, 128]))
        self.assertTrue(torch.allclose(tiled, full, atol=1e-6, rtol=1e-6))
