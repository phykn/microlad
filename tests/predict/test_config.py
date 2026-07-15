import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.predict import PredictConfig, load_predict_config


class PredictConfigTest(unittest.TestCase):
    predictor = SimpleNamespace(image_size=64, num_phases=3)

    def test_uses_training_size_when_scale_is_disabled(self):
        config = self._load(enabled=False)
        options = config.make_options(self.predictor)

        self.assertEqual(options.volume_size, 64)
        self.assertEqual(options.tile_overlap, 0.0)

    def test_uses_scale_settings_when_enabled(self):
        config = self._load(enabled=True)
        options = config.make_options(self.predictor)

        self.assertEqual(options.volume_size, 128)
        self.assertEqual(options.tile_overlap, 0.25)

    def test_rejects_non_boolean_scale_flag(self):
        config = self._load(enabled="yes")

        with self.assertRaisesRegex(ValueError, "scale.enabled"):
            config.make_options(self.predictor)

    @staticmethod
    def _load(enabled: bool | str) -> PredictConfig:
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "predict.yaml"
        flag = str(enabled).lower() if isinstance(enabled, bool) else f'"{enabled}"'
        path.write_text(
            "model:\n"
            "  run_dir: run/test\n"
            "generation: {}\n"
            "scale:\n"
            f"  enabled: {flag}\n"
            "  volume_size: 128\n"
            "  tile_overlap: 0.25\n",
            encoding="utf-8",
        )
        config = load_predict_config(path)
        tmp.cleanup()
        return config


if __name__ == "__main__":
    unittest.main()
