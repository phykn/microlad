import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.misc import require_finite, require_int, require_number, save_config


class ConfigTest(unittest.TestCase):
    def test_save_config_serializes_paths_and_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "data_dir": Path("data/train"),
                "phase_fractions": (0.25, 0.75),
            }

            save_config(tmp, cfg, name="model")
            values = yaml.safe_load(
                (Path(tmp) / "model.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(Path(values["data_dir"]), Path("data/train"))
        self.assertEqual(values["phase_fractions"], [0.25, 0.75])

    def test_save_config_preserves_section_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_config(
                tmp,
                {"data": {"data_dir": {0: Path("train/0")}}},
                name="model",
            )
            values = yaml.safe_load(
                (Path(tmp) / "model.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(
            Path(values["data"]["data_dir"][0]),
            Path("train/0"),
        )


class ScalarValidationTest(unittest.TestCase):
    def test_require_int_accepts_integer(self):
        require_int("value", 3)

    def test_require_int_rejects_non_integer_and_bool(self):
        for value in (1.5, "3", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    require_int("value", value)

    def test_require_number_accepts_real_scalar(self):
        require_number("value", 0.5)

    def test_require_number_rejects_non_real_and_bool(self):
        for value in ("0.5", True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "real scalar"):
                    require_number("value", value)

    def test_require_number_rejects_non_finite_value(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    require_number("value", value)


class TensorValidationTest(unittest.TestCase):
    def test_require_finite_accepts_finite_values(self):
        require_finite("values", torch.tensor([0.0, 1.0, -1.0]))

    def test_require_finite_rejects_nan_and_inf(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "values.*finite"):
                    require_finite("values", torch.tensor([value]))


if __name__ == "__main__":
    unittest.main()
