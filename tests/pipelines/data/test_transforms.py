import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from src.pipelines.data.transforms import augment_patch, crop_square, resize_patch


class DataTransformTest(unittest.TestCase):
    def test_crop_square_returns_requested_region(self):
        pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)

        with mock_patch("numpy.random.randint", side_effect=(1, 2)):
            cropped = crop_square(pixels, 2)

        np.testing.assert_array_equal(cropped, pixels[1:3, 2:4])

    def test_resize_patch_preserves_categorical_labels(self):
        pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        resized = resize_patch(pixels, 4)

        expected = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(resized, expected)
        self.assertEqual(resized.dtype, np.uint8)

    def test_resize_patch_returns_copy_when_size_is_unchanged(self):
        pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)

        resized = resize_patch(pixels, 2)

        self.assertFalse(np.shares_memory(resized, pixels))
        np.testing.assert_array_equal(resized, pixels)

    def test_crop_and_resize_require_integer_size(self):
        pixels = np.zeros((2, 2), dtype=np.uint8)

        for transform in (crop_square, resize_patch):
            with self.subTest(transform=transform.__name__):
                with self.assertRaisesRegex(ValueError, "integer"):
                    transform(pixels, 1.5)

    def test_transforms_reject_non_uint8_labels(self):
        pixels = np.zeros((2, 2), dtype=np.float32)

        for transform, args in (
            (crop_square, (pixels, 2)),
            (resize_patch, (pixels, 2)),
            (augment_patch, (pixels,)),
        ):
            with self.subTest(transform=transform.__name__):
                with self.assertRaisesRegex(ValueError, "uint8"):
                    transform(*args)

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
