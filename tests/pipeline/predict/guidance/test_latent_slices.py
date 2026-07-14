import unittest

import torch

from src.pipeline.predict.guidance.latent_slices import sample_slices


class LatentSliceSamplerTest(unittest.TestCase):
    def test_balanced_sampler_uses_all_axes(self):
        volume = torch.zeros(1, 1, 16, 16, 16)
        volume[:, :, 1:] = 1.0

        samples = sample_slices(volume, count=3, crop_size=16)

        self.assertEqual(samples.shape, torch.Size([3, 1, 16, 16]))

    def test_single_slice_sampler_cycles_axes_with_step_offset(self):
        depth = torch.arange(16).view(16, 1, 1) * 10_000
        row = torch.arange(16).view(1, 16, 1) * 100
        column = torch.arange(16).view(1, 1, 16)
        volume = (depth + row + column).float().view(1, 1, 16, 16, 16)

        increments = []
        for step in range(6):
            sampled = sample_slices(
                volume,
                count=1,
                crop_size=16,
                axis_offset=step % 3,
            )[0, 0]
            increments.append(
                (
                    int(sampled[1, 0] - sampled[0, 0]),
                    int(sampled[0, 1] - sampled[0, 0]),
                )
            )

        self.assertEqual(
            increments,
            [
                (100, 1),
                (10_000, 1),
                (10_000, 100),
                (100, 1),
                (10_000, 1),
                (10_000, 100),
            ],
        )


if __name__ == "__main__":
    unittest.main()
