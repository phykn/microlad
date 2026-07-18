import unittest

import torch

from src.predict.volume import merge_planes, slice_volume


class VolumeTest(unittest.TestCase):
    def test_planes_follow_zyx_axis_contract(self):
        z, y, x = torch.meshgrid(
            torch.arange(2),
            torch.arange(3),
            torch.arange(4),
            indexing="ij",
        )
        tagged = (100 * z + 10 * y + x).unsqueeze(0)

        axis_0 = slice_volume(tagged, 0)
        axis_1 = slice_volume(tagged, 1)
        axis_2 = slice_volume(tagged, 2)

        self.assertTrue(torch.equal(axis_0[1, 0], tagged[0, 1, :, :]))
        self.assertTrue(torch.equal(axis_1[2, 0], tagged[0, :, 2, :]))
        self.assertTrue(torch.equal(axis_2[3, 0], tagged[0, :, :, 3]))

    def test_plane_conversion_round_trips_every_axis(self):
        volume = torch.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)

        for axis in range(3):
            with self.subTest(axis=axis):
                planes = slice_volume(volume, axis)
                restored = merge_planes(planes, axis)
                self.assertTrue(torch.equal(restored, volume))


if __name__ == "__main__":
    unittest.main()
