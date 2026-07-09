import unittest

from src.common.validation import require_finite_number, require_int


class ScalarValidationTest(unittest.TestCase):
    def test_require_int_accepts_integer(self):
        require_int("value", 3)

    def test_require_int_rejects_non_integer_and_bool(self):
        for value in (1.5, "3", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    require_int("value", value)

    def test_require_finite_number_accepts_real_scalar(self):
        require_finite_number("value", 0.5)

    def test_require_finite_number_rejects_non_real_and_bool(self):
        for value in ("0.5", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "real scalar"):
                    require_finite_number("value", value)

    def test_require_finite_number_rejects_non_finite_value(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    require_finite_number("value", value)


if __name__ == "__main__":
    unittest.main()
