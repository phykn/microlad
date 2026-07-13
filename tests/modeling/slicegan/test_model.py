import unittest

import torch

from src.modeling.slicegan import (
    MIN_SLICE_SIZE,
    NOISE_CHANNELS,
    SliceGANCritic,
    SliceGANGenerator,
    output_size,
)


class LatentSliceGANModelTest(unittest.TestCase):
    def test_generator_and_critic_support_vae_latent_sizes(self):
        with torch.device("meta"):
            generator = SliceGANGenerator(latent_ch=8)
            critic = SliceGANCritic(latent_ch=8)

            for size in (16, 20, 24, 32):
                with self.subTest(size=size):
                    noise_size = size // 4
                    noise = torch.empty(
                        1,
                        NOISE_CHANNELS,
                        noise_size,
                        noise_size,
                        noise_size,
                        device="meta",
                    )
                    latent = generator(noise)
                    scores = critic(
                        torch.empty(1, 8, size, size, device="meta")
                    )

                    self.assertEqual(
                        latent.shape,
                        torch.Size([1, 8, size, size, size]),
                    )
                    self.assertGreater(scores.shape[-2], 0)
                    self.assertGreater(scores.shape[-1], 0)

    def test_output_size_uses_latent_scale(self):
        self.assertEqual(MIN_SLICE_SIZE, 16)
        self.assertEqual(output_size(4), 16)
        self.assertEqual(output_size(5), 20)
        self.assertEqual(output_size(8), 32)

    def test_generator_declares_tiled_render_scale(self):
        generator = SliceGANGenerator(8)

        self.assertEqual(generator.scale_factor, 4)


if __name__ == "__main__":
    unittest.main()
