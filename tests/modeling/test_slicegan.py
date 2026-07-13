import unittest

import torch

from src.modeling.slicegan import SliceGANGenerator, slicegan_output_size


class SliceGANModelTest(unittest.TestCase):
    def test_fully_convolutional_generator_preserves_voxel_scale(self):
        with torch.device("meta"):
            scalable = SliceGANGenerator(3, fully_convolutional=True)
            base = SliceGANGenerator(3)
            scalable_64 = scalable(torch.empty(1, 32, 4, 4, 4, device="meta"))
            scalable_128 = scalable(torch.empty(1, 32, 8, 8, 8, device="meta"))
            base_64 = base(torch.empty(1, 32, 4, 4, 4, device="meta"))

        self.assertEqual(scalable_64.shape, torch.Size([1, 3, 64, 64, 64]))
        self.assertEqual(scalable_128.shape, torch.Size([1, 3, 128, 128, 128]))
        self.assertEqual(base_64.shape, torch.Size([1, 3, 64, 64, 64]))

    def test_noise_grid_controls_output_size(self):
        self.assertEqual(slicegan_output_size(4), 64)
        self.assertEqual(slicegan_output_size(8), 128)
