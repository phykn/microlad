import unittest

import torch

from src.predict.slices import extract_slice, replace_slice


class PredictSlicesTest(unittest.TestCase):
    def test_extract_slice_rejects_invalid_axis(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "axis"):
            extract_slice(volume, axis=99, index=0)

    def test_replace_slice_rejects_invalid_axis(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "axis"):
            replace_slice(volume, axis=99, index=0, image=torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
