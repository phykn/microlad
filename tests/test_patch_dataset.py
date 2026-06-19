import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import tifffile

from src.dataset import PatchDataset


def save_grayscale(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


class PatchDatasetTest(unittest.TestCase):
    def test_len_is_source_image_count_and_getitem_returns_random_in_memory_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(
                image_dir / "sem_a.png", np.zeros((130, 130), dtype=np.uint8)
            )
            save_grayscale(
                image_dir / "sem_b.tif", np.full((64, 64), 200, dtype=np.uint8)
            )
            before_files = sorted(
                path.relative_to(image_dir)
                for path in image_dir.rglob("*")
                if path.is_file()
            )

            dataset = PatchDataset(image_dir, patch_size=64)

            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(first.shape, torch.Size([1, 64, 64]))
            self.assertEqual(first.dtype, torch.float32)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)

            second = dataset[1]
            self.assertTrue(
                torch.allclose(second, torch.full((1, 64, 64), 200 / 255.0))
            )

            after_files = sorted(
                path.relative_to(image_dir)
                for path in image_dir.rglob("*")
                if path.is_file()
            )
            self.assertEqual(after_files, before_files)

    def test_getitem_uses_index_to_choose_source_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem_a.png", np.zeros((96, 96), dtype=np.uint8))
            save_grayscale(
                image_dir / "sem_b.png", np.full((96, 96), 255, dtype=np.uint8)
            )

            dataset = PatchDataset(image_dir, patch_size=64)

            first = dataset[0]
            second = dataset[1]
            self.assertEqual(float(first.mean()), 0.0)
            self.assertEqual(float(second.mean()), 1.0)

    def test_exact_patch_size_image_always_uses_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(
                image_dir / "sem.tif", np.full((64, 64), 200, dtype=np.uint8)
            )

            dataset = PatchDataset(image_dir, patch_size=64)

            last = dataset[0]
            self.assertTrue(torch.allclose(last, torch.full((1, 64, 64), 200 / 255.0)))

    def test_raises_when_no_image_can_produce_a_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(
                image_dir / "too_small.png", np.zeros((63, 64), dtype=np.uint8)
            )

            with self.assertRaisesRegex(ValueError, "No 64x64 patches"):
                PatchDataset(image_dir, patch_size=64)

    def test_rgb_png_is_converted_to_grayscale_model_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(image_dir / "rgb.png")

            dataset = PatchDataset(image_dir, patch_size=64)

            patch = dataset[0]
            self.assertEqual(patch.shape, torch.Size([1, 64, 64]))
            self.assertTrue(torch.allclose(patch, torch.full((1, 64, 64), 128 / 255.0)))

    def test_uint16_tiff_is_scaled_to_uint8_before_model_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            tifffile.imwrite(
                image_dir / "sem.tiff", np.full((64, 64), 65535, dtype=np.uint16)
            )

            dataset = PatchDataset(image_dir, patch_size=64)

            patch = dataset[0]
            self.assertTrue(torch.allclose(patch, torch.ones((1, 64, 64))))

    def test_dhw_tiff_volume_returns_2d_slice_model_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            volume = np.full((3, 64, 64), 255, dtype=np.uint8)
            tifffile.imwrite(image_dir / "volume.tif", volume, photometric="minisblack")

            dataset = PatchDataset(image_dir, patch_size=64)

            patch = dataset[0]
            self.assertEqual(patch.shape, torch.Size([1, 64, 64]))
            self.assertTrue(torch.allclose(patch, torch.ones((1, 64, 64))))


if __name__ == "__main__":
    unittest.main()
