import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import load_axis_images


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)


class AxisImagesTest(unittest.TestCase):
    def test_loads_images_and_conditions_from_axis_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {axis: root / str(axis) for axis in range(3)}
            for axis, directory in dirs.items():
                _write_image(directory / f"{axis}.png")

            paths, conditions = load_axis_images(dirs)

        self.assertEqual(
            [path.parent.name for path in paths],
            ["0", "1", "2"],
        )
        self.assertEqual(conditions, (0, 1, 2))

    def test_requires_every_plane_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {axis: root / str(axis) for axis in range(2)}
            for axis, directory in dirs.items():
                _write_image(directory / f"{axis}.png")

            with self.assertRaisesRegex(ValueError, "exactly"):
                load_axis_images(dirs)

    def test_rejects_empty_plane_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {axis: root / str(axis) for axis in range(3)}
            for directory in dirs.values():
                directory.mkdir()
            _write_image(dirs[0] / "0.png")
            _write_image(dirs[1] / "1.png")

            with self.assertRaisesRegex(ValueError, "axis 2"):
                load_axis_images(dirs)


if __name__ == "__main__":
    unittest.main()
