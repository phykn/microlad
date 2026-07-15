import unittest

import torch

from src.predict.volume import merge_planes, slice_volume


class VolumeTest(unittest.TestCase):
    def test_plane_conversion_round_trips_every_axis(self):
        volume = torch.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)

        for axis in range(3):
            with self.subTest(axis=axis):
                planes = slice_volume(volume, axis)
                restored = merge_planes(planes, axis)
                self.assertTrue(torch.equal(restored, volume))


if __name__ == "__main__":
    unittest.main()
