import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data import AxisPatchDataset


class AxisPatchDatasetTest(unittest.TestCase):
    def test_same_image_can_have_each_axis_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase.png"
            Image.fromarray(
                np.array([[0, 1], [1, 0]], dtype=np.uint8)
            ).save(path)
            dataset = AxisPatchDataset(
                [path, path, path],
                [0, 1, 2],
                crop_size=2,
                image_size=2,
                num_phases=2,
            )

            samples = [dataset[index] for index in range(3)]

        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            [condition.item() for _, _, condition in samples],
            [0, 1, 2],
        )
        for image, fractions, condition in samples:
            self.assertEqual(image.shape, torch.Size([1, 2, 2]))
            self.assertTrue(torch.equal(fractions, torch.tensor([0.5, 0.5])))
            self.assertEqual(condition.dtype, torch.long)

    def test_requires_one_valid_condition_per_image(self):
        cases = (
            ([0, 1], "one condition"),
            ([0, 1, 3], "0, 1, or 2"),
            ([0, 1, True], "integer"),
        )

        for conditions, message in cases:
            with self.subTest(conditions=conditions):
                with self.assertRaisesRegex(ValueError, message):
                    AxisPatchDataset(
                        ["a.png", "b.png", "c.png"],
                        conditions,
                        crop_size=2,
                        image_size=2,
                        num_phases=2,
                    )


if __name__ == "__main__":
    unittest.main()
