import unittest

import numpy as np

from src.data.segment import segment_otsu


class MultiOtsuSegmentationTest(unittest.TestCase):
    def test_segment_otsu_returns_phase_labels(self):
        image = np.array(
            [
                [10, 12, 120, 122, 230, 232],
                [11, 13, 121, 123, 231, 233],
            ],
            dtype=np.uint8,
        )

        labels = segment_otsu(image, num_phases=3)

        expected = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(labels, expected)

    def test_segment_otsu_accepts_dhw_volume(self):
        volume = np.zeros((2, 4, 5), dtype=np.uint8)
        volume[:, :, :2] = 0
        volume[:, :, 2:4] = 100
        volume[:, :, 4:] = 200

        labels = segment_otsu(volume, num_phases=3)

        expected = np.zeros_like(volume)
        expected[:, :, 2:4] = 1
        expected[:, :, 4:] = 2
        np.testing.assert_array_equal(labels, expected)

    def test_segment_otsu_requires_2d_uint8_image(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            segment_otsu(np.zeros((1, 2, 3, 4), dtype=np.uint8), num_phases=2)

        with self.assertRaisesRegex(ValueError, "dtype"):
            segment_otsu(np.zeros((2, 3), dtype=np.float32), num_phases=2)

    def test_segment_otsu_requires_at_least_two_phases(self):
        image = np.zeros((2, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "num_phases"):
            segment_otsu(image, num_phases=1)

    def test_segment_otsu_requires_integer_phase_count_in_uint8_range(self):
        image = np.array([[0, 1]], dtype=np.uint8)

        for num_phases in (2.5, True, 257):
            with self.subTest(num_phases=num_phases):
                with self.assertRaisesRegex(ValueError, "num_phases"):
                    segment_otsu(image, num_phases=num_phases)

    def test_segment_otsu_requires_enough_distinct_values(self):
        image = np.array([[0, 0], [1, 1]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "distinct"):
            segment_otsu(image, num_phases=3)

    def test_segment_otsu_rejects_empty_images(self):
        image = np.empty((0, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            segment_otsu(image, num_phases=2)


if __name__ == "__main__":
    unittest.main()
