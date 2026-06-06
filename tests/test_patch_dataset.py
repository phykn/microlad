import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import PatchDataset


def save_grayscale(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


class PatchDatasetTest(unittest.TestCase):
    def test_reads_source_images_and_returns_in_memory_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem_a.png", np.arange(130 * 130, dtype=np.uint8).reshape(130, 130))
            save_grayscale(image_dir / "sem_b.tif", np.full((64, 64), 200, dtype=np.uint8))
            before_files = sorted(path.relative_to(image_dir) for path in image_dir.rglob("*") if path.is_file())

            dataset = PatchDataset(image_dir, patch_size=64, stride=64)

            self.assertEqual(len(dataset), 5)
            first = dataset[0]
            self.assertEqual(set(first), {"image", "source_path", "xy"})
            self.assertEqual(first["image"].shape, torch.Size([1, 64, 64]))
            self.assertEqual(first["image"].dtype, torch.float32)
            self.assertGreaterEqual(float(first["image"].min()), 0.0)
            self.assertLessEqual(float(first["image"].max()), 1.0)
            self.assertEqual(Path(first["source_path"]).name, "sem_a.png")
            self.assertEqual(first["xy"], (0, 0))

            last = dataset[4]
            self.assertEqual(Path(last["source_path"]).name, "sem_b.tif")
            self.assertEqual(last["xy"], (0, 0))
            self.assertTrue(torch.allclose(last["image"], torch.full((1, 64, 64), 200 / 255.0)))

            after_files = sorted(path.relative_to(image_dir) for path in image_dir.rglob("*") if path.is_file())
            self.assertEqual(after_files, before_files)

    def test_uses_stride_to_index_overlapping_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem.png", np.zeros((96, 96), dtype=np.uint8))

            dataset = PatchDataset(image_dir, patch_size=64, stride=32)

            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                [dataset[index]["xy"] for index in range(len(dataset))],
                [(0, 0), (32, 0), (0, 32), (32, 32)],
            )

    def test_raises_when_no_image_can_produce_a_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "too_small.png", np.zeros((63, 64), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "No 64x64 patches"):
                PatchDataset(image_dir, patch_size=64, stride=64)


if __name__ == "__main__":
    unittest.main()
