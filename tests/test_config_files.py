from pathlib import Path
from types import SimpleNamespace
import unittest

from src.misc import load_config
from src.predict import MPDDOptions, load_predict_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigFileTest(unittest.TestCase):
    def test_training_config_has_runtime_inputs(self):
        config = load_config(ROOT / "config" / "model.yaml")
        required = {
            "data_dir",
            "crop_size",
            "size",
            "num_phases",
            "base_ch",
            "time_dim",
            "timesteps",
            "beta_start",
            "beta_end",
            "lr",
            "steps",
            "run_root",
        }

        self.assertFalse(required - config.keys())

    def test_prediction_config_builds_public_options(self):
        config = load_predict_config(ROOT / "config" / "predict.yaml")
        predictor = SimpleNamespace(image_size=64, num_phases=3)
        options = config.make_options(predictor)

        self.assertTrue(str(config.run_dir))
        self.assertIsInstance(options, MPDDOptions)


if __name__ == "__main__":
    unittest.main()
