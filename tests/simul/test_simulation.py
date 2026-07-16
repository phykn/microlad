import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from src.data import load_axis_manifest
from src.simul import save_simulation


class SimulationExportTest(unittest.TestCase):
    def test_saves_all_planes_with_exact_orientation_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            volume_paths, plane_paths = save_simulation(
                root,
                count=1,
                geometry={
                    "mode": "dry",
                    "size": 20,
                    "big_radius": 3,
                    "big_fraction": 0.15,
                    "small_fraction": 0.05,
                    "shape": "aligned_ellipsoid",
                    "elongation": 2.0,
                    "alignment_axis": "x",
                },
                export={"planes": ["yz", "xy", "xz"], "trim": 0},
            )
            volume = tifffile.imread(volume_paths[0])
            manifest = json.loads((root / "manifest.json").read_text())
            dataset_paths, conditions = load_axis_manifest(root / "manifest.json")

            with Image.open(plane_paths["xy"][3]) as image:
                xy = np.asarray(image)
                mode = image.mode
            with Image.open(plane_paths["xz"][5]) as image:
                xz = np.asarray(image)
            with Image.open(plane_paths["yz"][7]) as image:
                yz = np.asarray(image)
            plane_histograms = {}
            for plane, paths in plane_paths.items():
                actual = np.zeros(3, dtype=np.int64)
                for path in paths:
                    with Image.open(path) as image:
                        actual += np.bincount(
                            np.asarray(image).ravel(),
                            minlength=3,
                        )
                plane_histograms[plane] = actual

        self.assertEqual(set(plane_paths), {"xy", "xz", "yz"})
        self.assertTrue(all(len(paths) == 20 for paths in plane_paths.values()))
        self.assertEqual(len(dataset_paths), 60)
        self.assertEqual(
            np.bincount(conditions, minlength=3).tolist(),
            [20, 20, 20],
        )
        self.assertEqual(mode, "P")
        np.testing.assert_array_equal(xy, volume[3, :, :])
        np.testing.assert_array_equal(xz, volume[:, 5, :])
        np.testing.assert_array_equal(yz, volume[:, :, 7])

        expected_histogram = np.bincount(volume.ravel(), minlength=3)
        for actual in plane_histograms.values():
            np.testing.assert_array_equal(actual, expected_histogram)

        record = manifest["volumes"][0]
        np.testing.assert_allclose(
            record["achieved_fractions"],
            expected_histogram / volume.size,
        )
        self.assertEqual(manifest["geometry"]["shape"], "aligned_ellipsoid")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["volume_axes"], "ZYX")
        self.assertEqual(
            manifest["axis_sources"],
            {
                "xy": "train/xy",
                "xz": "train/xz",
                "yz": "train/yz",
            },
        )
        self.assertEqual(manifest["volumes"][0]["path"], "gt/volume.tif")
        self.assertEqual(manifest["export"]["planes"], ["yz", "xy", "xz"])

    def test_rejects_unknown_or_duplicate_planes(self):
        settings = {
            "mode": "dry",
            "size": 16,
            "big_radius": 3,
            "big_fraction": 0.15,
            "small_fraction": 0.05,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cases = (
                (["xy", "xy"], "duplicates"),
                (["xy", "xz", "zx"], "unknown"),
                (["xy", "xz"], "include yz"),
            )
            for planes, message in cases:
                with self.subTest(planes=planes):
                    with self.assertRaisesRegex(ValueError, message):
                        save_simulation(
                            Path(tmp) / str(len(planes)) / planes[0],
                            count=1,
                            geometry=settings,
                            export={"planes": planes},
                        )


if __name__ == "__main__":
    unittest.main()
