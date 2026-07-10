import unittest

import torch

from src.common.tensors.validation import require_finite


class PredictValidationTest(unittest.TestCase):
    def test_require_finite_accepts_finite_values(self):
        require_finite("values", torch.tensor([0.0, 1.0, -1.0]))

    def test_require_finite_rejects_nan_and_inf(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "values.*finite"):
                    require_finite("values", torch.tensor([value]))


if __name__ == "__main__":
    unittest.main()
