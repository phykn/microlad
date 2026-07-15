import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from simul.gen_data import (
    generate_sphere_dataset,
    generate_sphere_datasets,
    generate_sphere_geometry,
    generate_sphere_volume,
)
from src.data import PatchDataset


class SphereDataGeneratorTest(unittest.TestCase):
    def test_volume_is_reproducible_and_spheres_do_not_overlap(self):
        settings = {
            "size": 32,
            "big_radius": 4,
            "big_fraction": 0.25,
            "small_fraction": 0.1,
            "seed": 7,
        }
        first, spheres = generate_sphere_geometry(**settings)
        second = generate_sphere_volume(**settings)

        self.assertEqual(first.shape, (32, 32, 32))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(sorted(np.unique(first).tolist()), [0, 1, 2])
        np.testing.assert_array_equal(first, second)
        for index, (center, radius, label) in enumerate(spheres):
            self.assertTrue(np.all(center >= radius))
            self.assertTrue(np.all(center < settings["size"] - radius))
            for other_center, other_radius, _ in spheres[index + 1 :]:
                distance_squared = np.sum((center - other_center) ** 2)
                self.assertGreater(
                    distance_squared,
                    (radius + other_radius) ** 2,
                )
            if label == 2 and center[0] != settings["size"] - radius - 1:
                supports = []
                for other_center, other_radius, _ in spheres[:index]:
                    horizontal_squared = np.sum(
                        (center[1:] - other_center[1:]) ** 2
                    )
                    minimum_distance_squared = (radius + other_radius) ** 2
                    if horizontal_squared > minimum_distance_squared:
                        continue
                    vertical_clearance = int(
                        np.floor(
                            np.sqrt(
                                minimum_distance_squared - horizontal_squared
                            )
                        )
                    ) + 1
                    supports.append(
                        center[0] == other_center[0] - vertical_clearance
                    )
                self.assertTrue(any(supports))
        big_z = [int(center[0]) for center, _, label in spheres if label == 2]
        self.assertGreaterEqual(len(set(big_z)), 8)
        z_occupancy = (first != 0).mean(axis=(1, 2))
        z_bins = np.asarray(
            [chunk.mean() for chunk in np.array_split(z_occupancy, 4)]
        )
        self.assertGreater(float(z_bins.min()), 0.19)
        self.assertLess(float(np.ptp(z_bins)), 0.1)
        self.assertGreater(float((first[8:24] == 1).mean()), 0.09)
        self.assertTrue(
            all(
                radius == settings["big_radius"] // 2
                for _, radius, label in spheres
                if label == 1
            )
        )

    def test_saves_gt_tiff_and_train_pngs_that_the_dataset_can_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            volume_path, slice_paths = generate_sphere_dataset(
                data,
                size=32,
                big_radius=4,
                big_fraction=0.25,
                small_fraction=0.1,
                seed=7,
            )
            volume = tifffile.imread(volume_path)
            with Image.open(data / "train" / "slice_z_016.png") as image:
                labels = np.asarray(image)
                mode = image.mode
            dataset = PatchDataset(
                slice_paths,
                crop_size=32,
                image_size=32,
                num_phases=3,
                segment=False,
            )
            patch, fractions = dataset[8]

        self.assertEqual(volume_path, data / "gt" / "volume.tif")
        self.assertEqual(len(slice_paths), 16)
        self.assertEqual(slice_paths[0].name, "slice_z_008.png")
        self.assertEqual(slice_paths[-1].name, "slice_z_023.png")
        self.assertFalse((data / "train" / "slice_z_000.png").exists())
        self.assertFalse((data / "train" / "slice_z_031.png").exists())
        self.assertEqual(mode, "P")
        np.testing.assert_array_equal(labels, volume[16])
        np.testing.assert_array_equal(patch[0].numpy(), volume[16])
        self.assertAlmostEqual(float(fractions.sum()), 1.0)

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            train = data / "train"
            train.mkdir(parents=True)
            (train / "keep.txt").write_text("keep")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                generate_sphere_dataset(
                    data,
                    size=32,
                    big_radius=4,
                    big_fraction=0.25,
                    small_fraction=0.1,
                    seed=7,
                )

    def test_saves_multiple_volumes_with_independent_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            volume_paths, slice_paths = generate_sphere_datasets(
                data,
                num_volumes=2,
                size=32,
                big_radius=4,
                big_fraction=0.25,
                small_fraction=0.1,
                seed=7,
            )
            first = tifffile.imread(volume_paths[0])
            second = tifffile.imread(volume_paths[1])

        self.assertEqual(
            [path.name for path in volume_paths],
            ["volume_000.tif", "volume_001.tif"],
        )
        self.assertEqual(len(slice_paths), 32)
        self.assertEqual(slice_paths[0].name, "volume_000_z_008.png")
        self.assertEqual(slice_paths[-1].name, "volume_001_z_023.png")
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
