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
            pixels = np.linspace(0, 4095, num=20, dtype=np.uint16).reshape(4, 5)
            Image.fromarray(pixels).save(path)

            image = load_image(path)

        self.assertEqual(image.shape, (4, 5))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image.min()), 0)
        self.assertEqual(int(image.max()), 255)


if __name__ == "__main__":
    unittest.main()
