import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import PatchDataset, SliceConditionDataset


def save_grayscale(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


class PatchDatasetTest(unittest.TestCase):
    def test_len_is_source_image_count_and_getitem_returns_random_in_memory_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem_a.png", np.zeros((130, 130), dtype=np.uint8))
            save_grayscale(image_dir / "sem_b.tif", np.full((64, 64), 200, dtype=np.uint8))
            before_files = sorted(path.relative_to(image_dir) for path in image_dir.rglob("*") if path.is_file())

            dataset = PatchDataset(image_dir, patch_size=64, seed=0)

            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(first.shape, torch.Size([1, 64, 64]))
            self.assertEqual(first.dtype, torch.float32)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 1.0)

            second = dataset[1]
            self.assertTrue(torch.allclose(second, torch.full((1, 64, 64), 200 / 255.0)))

            after_files = sorted(path.relative_to(image_dir) for path in image_dir.rglob("*") if path.is_file())
            self.assertEqual(after_files, before_files)

    def test_getitem_uses_index_to_choose_source_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem_a.png", np.zeros((96, 96), dtype=np.uint8))
            save_grayscale(image_dir / "sem_b.png", np.full((96, 96), 255, dtype=np.uint8))

            dataset = PatchDataset(image_dir, patch_size=64, seed=0)

            first = dataset[0]
            second = dataset[1]
            self.assertEqual(float(first.mean()), 0.0)
            self.assertEqual(float(second.mean()), 1.0)

    def test_exact_patch_size_image_always_uses_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "sem.tif", np.full((64, 64), 200, dtype=np.uint8))

            dataset = PatchDataset(image_dir, patch_size=64, seed=0)

            last = dataset[0]
            self.assertTrue(torch.allclose(last, torch.full((1, 64, 64), 200 / 255.0)))

    def test_raises_when_no_image_can_produce_a_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            save_grayscale(image_dir / "too_small.png", np.zeros((63, 64), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "No 64x64 patches"):
                PatchDataset(image_dir, patch_size=64)


class SliceConditionDatasetTest(unittest.TestCase):
    def test_returns_full_condition_slice_and_3d_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            pixels = np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) % 256
            save_grayscale(image_dir / "sem.png", pixels.astype(np.uint8))

            dataset = SliceConditionDataset(image_dir, patch_size=64, axis="z", slice_index=12, seed=0)

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(set(sample), {"target", "condition", "axis", "slice_index"})
            self.assertEqual(sample["target"].shape, torch.Size([1, 64, 64]))
            self.assertEqual(sample["condition"].shape, torch.Size([1, 64, 64]))
            self.assertEqual(sample["target"].dtype, torch.float32)
            self.assertEqual(sample["condition"].dtype, torch.float32)
            self.assertEqual(sample["axis"], 0)
            self.assertEqual(sample["slice_index"], 12)
            self.assertTrue(torch.allclose(sample["condition"], sample["target"]))


if __name__ == "__main__":
    unittest.main()
