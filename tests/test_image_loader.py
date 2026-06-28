import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.io.image import load_image


class ImageLoaderTest(unittest.TestCase):
    def test_load_image_returns_grayscale_uint8_hw_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            rgb = np.zeros((4, 5, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            Image.fromarray(rgb, mode="RGB").save(path)

            image = load_image(path)

        self.assertEqual(image.shape, (4, 5))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image[0, 0]), 76)

    def test_load_image_scales_uint16_tiff_to_uint8(self):
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

            image = load_image(path)

        self.assertEqual(image.shape, (4, 5))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image.min()), 0)
        self.assertEqual(int(image.max()), 255)
        self.assertEqual(image[0].tolist(), [0, 63, 127, 191, 255])

    def test_load_image_preserves_low_valued_uint16_phase_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tif"
            pixels = np.array([[0, 1], [1, 0]], dtype=np.uint16)
            Image.fromarray(pixels).save(path)

            image = load_image(path)

        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, pixels.astype(np.uint8))

    def test_load_image_converts_palette_images_to_grayscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "palette.png"
            image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="P")
            palette = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
            image.putpalette(palette)
            image.save(path)

            loaded = load_image(path)

        self.assertEqual(loaded.tolist(), [[0, 255]])


if __name__ == "__main__":
    unittest.main()
