import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.image import load_gray, load_labels


class ImageLoaderTest(unittest.TestCase):
    def test_load_gray_returns_uint8_hw_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            rgb = np.zeros((4, 5, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            Image.fromarray(rgb, mode="RGB").save(path)

            image = load_gray(path)

        self.assertEqual(image.shape, (4, 5))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image[0, 0]), 76)

    def test_load_gray_scales_uint16_tiff_to_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.tif"
            pixels = np.array(
                [
                    [0, 1024, 2048, 3072, 4095],
                    [0, 1024, 2048, 3072, 4095],
                    [0, 1024, 2048, 3072, 4095],
                    [0, 1024, 2048, 3072, 4095],
                ],
                dtype=np.uint16,
            )
            Image.fromarray(pixels).save(path)

            image = load_gray(path)

        self.assertEqual(image.shape, (4, 5))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image.min()), 0)
        self.assertEqual(int(image.max()), 255)
        self.assertEqual(image[0].tolist(), [0, 63, 127, 191, 255])

    def test_load_gray_scales_normalized_float_to_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normalized.tif"
            pixels = np.array(
                [
                    [0.0, 0.5, 1.0],
                    [0.0, 0.5, 1.0],
                    [0.0, 0.5, 1.0],
                ],
                dtype=np.float32,
            )
            Image.fromarray(pixels).save(path)

            image = load_gray(path)

        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[0].tolist(), [0, 127, 255])

    def test_load_gray_rejects_non_finite_values(self):
        cases = [
            np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, np.inf], [1.0, 0.0]], dtype=np.float32),
        ]
        for pixels in cases:
            with self.subTest(pixels=pixels):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "invalid.tif"
                    Image.fromarray(pixels).save(path)

                    with self.assertRaisesRegex(ValueError, "finite"):
                        load_gray(path)

    def test_load_gray_rounds_near_integer_float_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "near-labels.tif"
            pixels = np.array(
                [
                    [0.0, 0.99999994],
                    [0.99999994, 0.0],
                ],
                dtype=np.float32,
            )
            Image.fromarray(pixels).save(path)

            image = load_gray(path)

        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(
            image,
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
        )

    def test_load_gray_preserves_low_uint16_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tif"
            pixels = np.array([[0, 1], [1, 0]], dtype=np.uint16)
            Image.fromarray(pixels).save(path)

            image = load_gray(path)

        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, pixels.astype(np.uint8))

    def test_load_gray_converts_palette_to_grayscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "palette.png"
            image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="P")
            palette = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
            image.putpalette(palette)
            image.save(path)

            loaded = load_gray(path)

        self.assertEqual(loaded.tolist(), [[0, 255]])

    def test_load_labels_preserves_palette_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.png"
            image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="P")
            palette = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
            image.putpalette(palette)
            image.save(path)

            loaded = load_labels(path)

        np.testing.assert_array_equal(loaded, np.array([[0, 1]], dtype=np.uint8))

    def test_load_labels_uses_first_channel_of_rgb_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            first_channel = np.array([[0, 1], [2, 1]], dtype=np.uint8)
            pixels = np.stack(
                [
                    first_channel,
                    np.full_like(first_channel, 17),
                    np.full_like(first_channel, 29),
                ],
                axis=-1,
            )
            Image.fromarray(pixels, mode="RGB").save(path)

            loaded = load_labels(path)

        np.testing.assert_array_equal(loaded, first_channel)

    def test_load_labels_rejects_non_integer_float_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intensity.tif"
            pixels = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
            Image.fromarray(pixels).save(path)

            with self.assertRaisesRegex(ValueError, "integer"):
                load_labels(path)

    def test_load_labels_accepts_near_integer_float_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tif"
            pixels = np.array([[0.0, 0.99999994]], dtype=np.float32)
            Image.fromarray(pixels).save(path)

            loaded = load_labels(path)

        np.testing.assert_array_equal(loaded, np.array([[0, 1]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
