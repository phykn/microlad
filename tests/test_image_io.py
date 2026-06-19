import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from src.helper import load_image


class ImageIOTest(unittest.TestCase):
    def test_rgb_tiff_becomes_grayscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.tif"
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            tifffile.imwrite(path, rgb, photometric="rgb")

            image = load_image(path)

            self.assertEqual(image.shape, (64, 64))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image[0, 0]), 76)

    def test_uint16_tiff_uses_data_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sem.tiff"
            pixels = np.linspace(0, 4095, num=64 * 64, dtype=np.uint16).reshape(64, 64)
            tifffile.imwrite(path, pixels)

            image = load_image(path)

            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image.min()), 0)
            self.assertEqual(int(image.max()), 255)

    def test_tiff_stack_keeps_dhw_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "volume.tif"
            volume = np.stack(
                [
                    np.full((5, 6), 0, dtype=np.uint16),
                    np.full((5, 6), 1000, dtype=np.uint16),
                    np.full((5, 6), 2000, dtype=np.uint16),
                ],
                axis=0,
            )
            tifffile.imwrite(path, volume, photometric="minisblack")

            image = load_image(path)

            self.assertEqual(image.shape, (3, 5, 6))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image[0].mean()), 0)
            self.assertGreater(int(image[1].mean()), 0)
            self.assertEqual(int(image[2].mean()), 255)

    def test_rgb_tiff_stack_becomes_dhw_grayscale_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb_volume.tif"
            volume = np.zeros((3, 5, 6, 3), dtype=np.uint8)
            volume[..., 0] = 255
            tifffile.imwrite(path, volume, photometric="rgb")

            image = load_image(path)

            self.assertEqual(image.shape, (3, 5, 6))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image[0, 0, 0]), 76)


if __name__ == "__main__":
    unittest.main()
