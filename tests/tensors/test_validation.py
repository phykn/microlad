import unittest

import torch

from src.common.tensors.validation import validate_finite_tensor


class PredictValidationTest(unittest.TestCase):
    def test_validate_finite_tensor_accepts_finite_values(self):
        validate_finite_tensor("values", torch.tensor([0.0, 1.0, -1.0]))

    def test_validate_finite_tensor_rejects_nan_and_inf(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "values.*finite"):
                    validate_finite_tensor("values", torch.tensor([value]))


if __name__ == "__main__":
    unittest.main()
