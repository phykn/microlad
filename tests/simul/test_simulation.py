import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from src.data import load_axis_images
from src.simul import save_simulation


class SimulationExportTest(unittest.TestCase):
    def test_saves_all_axes_with_exact_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            volume_paths, axis_paths = save_simulation(
                root,
                count=1,
                geometry={
                    "size": 20,
                    "big_radius": 3,
                    "small_radius": 2,
                    "big_fraction": 0.15,
                    "small_fraction": 0.05,
                    "big_elongation": 2.0,
                },
                axes=[2, 0, 1],
            )
            volume = tifffile.imread(volume_paths[0])
            dirs = {axis: root / "train" / str(axis) for axis in range(3)}
            dataset_paths, conditions = load_axis_images(dirs)

            with Image.open(axis_paths[0][3]) as image:
                axis_0 = np.asarray(image)
                mode = image.mode
            with Image.open(axis_paths[1][5]) as image:
                axis_1 = np.asarray(image)
            with Image.open(axis_paths[2][7]) as image:
                axis_2 = np.asarray(image)
            histograms = {}
            for axis, paths in axis_paths.items():
                actual = np.zeros(3, dtype=np.int64)
                for path in paths:
                    with Image.open(path) as image:
                        actual += np.bincount(
                            np.asarray(image).ravel(),
                            minlength=3,
                        )
                histograms[axis] = actual

        self.assertEqual(set(axis_paths), {0, 1, 2})
        self.assertTrue(all(len(paths) == 20 for paths in axis_paths.values()))
        self.assertEqual(len(dataset_paths), 60)
        self.assertEqual(
            np.bincount(conditions, minlength=3).tolist(),
            [20, 20, 20],
        )
        self.assertEqual(mode, "P")
        np.testing.assert_array_equal(axis_0, volume[3, :, :])
        np.testing.assert_array_equal(axis_1, volume[:, 5, :])
        np.testing.assert_array_equal(axis_2, volume[:, :, 7])

        expected_histogram = np.bincount(volume.ravel(), minlength=3)
        for actual in histograms.values():
            np.testing.assert_array_equal(actual, expected_histogram)

        self.assertFalse(any(root.glob("*.json")))

    def test_rejects_unknown_or_duplicate_axes(self):
        settings = {
            "size": 16,
            "big_radius": 3,
            "small_radius": 2,
            "big_fraction": 0.15,
            "small_fraction": 0.05,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cases = (
                ([0, 0, 2], "duplicates"),
                ([0, 1, 3], "exactly"),
                ([0, 1], "exactly"),
            )
            for axes, message in cases:
                with self.subTest(axes=axes):
                    with self.assertRaisesRegex(ValueError, message):
                        save_simulation(
                            Path(tmp) / str(len(axes)) / str(axes[0]),
                            count=1,
                            geometry=settings,
                            axes=axes,
                        )

    def test_multiple_volumes_are_independent_random_arrangements(self):
        settings = {
            "size": 20,
            "big_radius": 3,
            "small_radius": 2,
            "big_fraction": 0.15,
            "small_fraction": 0.05,
        }
        with tempfile.TemporaryDirectory() as tmp:
            volume_paths, _ = save_simulation(
                tmp,
                count=2,
                geometry=settings,
            )
            first = tifffile.imread(volume_paths[0])
            second = tifffile.imread(volume_paths[1])

        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
