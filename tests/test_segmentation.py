import unittest

import numpy as np

from src.segment import segment_multi_otsu


class MultiOtsuSegmentationTest(unittest.TestCase):
    def test_segment_multi_otsu_returns_phase_labels(self):
        image = np.array(
            [
                [10, 12, 120, 122, 230, 232],
                [11, 13, 121, 123, 231, 233],
            ],
            dtype=np.uint8,
        )

        labels = segment_multi_otsu(image, num_phases=3)

        expected = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(labels, expected)

    def test_segment_multi_otsu_accepts_dhw_volume(self):
        volume = np.zeros((2, 4, 5), dtype=np.uint8)
        volume[:, :, :2] = 0
        volume[:, :, 2:4] = 100
        volume[:, :, 4:] = 200

        labels = segment_multi_otsu(volume, num_phases=3)

        expected = np.zeros_like(volume)
        expected[:, :, 2:4] = 1
        expected[:, :, 4:] = 2
        np.testing.assert_array_equal(labels, expected)

    def test_segment_multi_otsu_requires_2d_uint8_image(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            segment_multi_otsu(np.zeros((1, 2, 3, 4), dtype=np.uint8), num_phases=2)

        with self.assertRaisesRegex(ValueError, "dtype"):
            segment_multi_otsu(np.zeros((2, 3), dtype=np.float32), num_phases=2)

    def test_segment_multi_otsu_requires_at_least_two_phases(self):
        image = np.zeros((2, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "num_phases"):
            segment_multi_otsu(image, num_phases=1)


if __name__ == "__main__":
    unittest.main()
