import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from src.data.transforms import augment_patch


class DataAugmentTest(unittest.TestCase):
    def test_augment_patch_preserves_square_2d_shape_and_values(self):
        pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        for transform in range(8):
            with self.subTest(transform=transform):
                with mock_patch("numpy.random.randint", return_value=transform):
                    augmented = augment_patch(pixels)

                self.assertEqual(augmented.shape, pixels.shape)
                self.assertEqual(sorted(augmented.reshape(-1).tolist()), [0, 1, 2, 3])

    def test_augment_patch_rejects_non_2d_or_non_square_inputs(self):
        for invalid in (
            np.zeros((1, 2, 3), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "square"):
                    augment_patch(invalid)


if __name__ == "__main__":
    unittest.main()
