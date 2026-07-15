import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data import PatchDataset


def save_image(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8)).save(path)


class PatchDatasetTest(unittest.TestCase):
    def test_requires_integer_sizes_and_phase_count(self):
        cases = (
            {"crop_size": 4.5, "image_size": 4, "num_phases": 2},
            {"crop_size": 4, "image_size": 4.5, "num_phases": 2},
            {"crop_size": 4, "image_size": 4, "num_phases": 2.5},
        )

        for options in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "integer"):
                    PatchDataset(["unused.png"], **options)

    def test_phase_image_returns_float_tensor_with_phase_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array(
                [
                    [0, 1, 2, 0],
                    [1, 2, 0, 1],
                    [2, 0, 1, 2],
                    [0, 1, 2, 0],
                ],
                dtype=np.uint8,
            )
            save_image(image_dir / "phase.png", pixels)

            dataset = PatchDataset(
                [image_dir / "phase.png"],
                crop_size=4,
                image_size=4,
                num_phases=3,
                segment=False,
            )
            patch, fractions = dataset[0]

        expected = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))
        self.assertEqual(patch.dtype, torch.float32)
        self.assertTrue(torch.allclose(patch, expected))
        self.assertTrue(torch.equal(fractions, torch.tensor([0.375, 0.3125, 0.3125])))

    def test_returns_phase_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase.png"
            save_image(path, np.array([[0, 0], [1, 2]], dtype=np.uint8))
            dataset = PatchDataset(
                [path],
                crop_size=2,
                image_size=2,
                num_phases=3,
            )

            patch, fractions = dataset[0]

        self.assertEqual(patch.shape, torch.Size([1, 2, 2]))
        self.assertTrue(torch.equal(fractions, torch.tensor([0.5, 0.25, 0.25])))

    def test_gray_image_can_be_segmented_to_phase_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array(
                [
                    [0, 0, 100, 100],
                    [0, 0, 100, 100],
                    [200, 200, 100, 100],
                    [200, 200, 0, 0],
                ],
                dtype=np.uint8,
            )
            save_image(image_dir / "gray.png", pixels)

            dataset = PatchDataset(
                [image_dir / "gray.png"],
                crop_size=4,
                image_size=4,
                num_phases=3,
                segment=True,
            )
            patch, _ = dataset[0]

        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))
        self.assertEqual(patch.dtype, torch.float32)
        self.assertEqual(sorted(torch.unique(patch).tolist()), [0.0, 1.0, 2.0])

    def test_normalized_float_gray_image_can_be_segmented(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array(
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5, 0.5],
                    [1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            Image.fromarray(pixels).save(image_dir / "gray-float.tif")

            dataset = PatchDataset(
                [image_dir / "gray-float.tif"],
                crop_size=4,
                image_size=4,
                num_phases=3,
                segment=True,
            )
            patch, _ = dataset[0]

        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))
        self.assertEqual(sorted(torch.unique(patch).tolist()), [0.0, 1.0, 2.0])

    def test_segment_false_rejects_normalized_float_intensity_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array(
                [
                    [0.0, 0.5],
                    [1.0, 0.5],
                ],
                dtype=np.float32,
            )
            Image.fromarray(pixels).save(image_dir / "gray-float.tif")

            dataset = PatchDataset(
                [image_dir / "gray-float.tif"],
                crop_size=2,
                image_size=2,
                num_phases=3,
                segment=False,
            )

            with self.assertRaisesRegex(ValueError, "integer"):
                dataset[0]

    def test_four_phase_image_keeps_phase_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            save_image(image_dir / "phase.png", pixels)

            dataset = PatchDataset(
                [image_dir / "phase.png"],
                crop_size=2,
                image_size=2,
                num_phases=4,
                segment=False,
            )
            patch, _ = dataset[0]

        expected = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
        self.assertTrue(torch.allclose(patch, expected))

    def test_rejects_num_phases_that_exceed_uint8_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            save_image(image_dir / "phase.png", np.array([[0, 255]], dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "num_phases"):
                PatchDataset(
                    [image_dir / "phase.png"],
                    crop_size=1,
                    image_size=1,
                    num_phases=257,
                    segment=False,
                )

    def test_phase_image_rejects_values_outside_phase_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            save_image(image_dir / "phase.png", np.array([[0, 1], [2, 3]]))

            with self.assertRaisesRegex(ValueError, "phase"):
                dataset = PatchDataset(
                    [image_dir / "phase.png"],
                    crop_size=2,
                    image_size=2,
                    num_phases=3,
                    segment=False,
                )
                dataset[0]

    def test_crop_size_and_output_size_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            save_image(image_dir / "phase.png", np.zeros((8, 8), dtype=np.uint8))

            dataset = PatchDataset(
                [image_dir / "phase.png"],
                crop_size=8,
                image_size=4,
                num_phases=2,
                segment=False,
            )
            patch, _ = dataset[0]

        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))

    def test_rejects_images_smaller_than_crop_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            save_image(image_dir / "phase.png", np.zeros((4, 4), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "8x8 crop"):
                dataset = PatchDataset(
                    [image_dir / "phase.png"],
                    crop_size=8,
                    image_size=4,
                    num_phases=2,
                    segment=False,
                )
                dataset[0]

    def test_getitem_loads_only_requested_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.zeros((4, 4), dtype=np.uint8)
            save_image(image_dir / "phase.png", pixels)

            dataset = PatchDataset(
                [image_dir / "phase.png", image_dir / "missing.png"],
                crop_size=4,
                image_size=4,
                num_phases=2,
                segment=False,
            )
            patch, _ = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))

    def test_augmentation_preserves_shape_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            pixels = np.array(
                [
                    [0, 1, 2, 0],
                    [1, 2, 0, 1],
                    [2, 0, 1, 2],
                    [0, 1, 2, 0],
                ],
                dtype=np.uint8,
            )
            save_image(image_dir / "phase.png", pixels)

            dataset = PatchDataset(
                [image_dir / "phase.png"],
                crop_size=4,
                image_size=4,
                num_phases=3,
                segment=False,
                augment=True,
            )
            patch, _ = dataset[0]

        self.assertEqual(patch.shape, torch.Size([1, 4, 4]))
        self.assertEqual(sorted(torch.unique(patch).tolist()), [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
