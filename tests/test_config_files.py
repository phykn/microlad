from pathlib import Path
from types import SimpleNamespace
import unittest

from src.misc import load_mapping
from src.predict import MPDDOptions, load_predict_config
from src.train import load_train_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigFileTest(unittest.TestCase):
    def test_training_config_has_runtime_inputs(self):
        config = load_train_config(ROOT / "config" / "model.yaml")

        self.assertEqual(
            config.data.data_dir,
            {
                axis: (ROOT / "data" / "generated" / "train" / str(axis)).resolve()
                for axis in range(3)
            },
        )
        self.assertEqual(config.model.base_ch, 64)
        self.assertEqual(config.diffusion.timesteps, 1000)
        self.assertGreater(config.training.steps, 0)
        self.assertEqual(config.output.run_root, "run")

    def test_simulation_config_uses_one_geometry_generator(self):
        cfg = load_mapping(ROOT / "config" / "simul.yaml")
        geo = cfg["geometry"]

        self.assertEqual(
            set(geo),
            {
                "size",
                "big_radius",
                "small_radius",
                "big_fraction",
                "small_fraction",
                "big_elongation",
            },
        )
        self.assertGreaterEqual(geo["big_elongation"], 1.0)
        self.assertLessEqual(geo["big_elongation"], 4.0)
        total = geo["big_fraction"] + geo["small_fraction"]
        self.assertGreater(total, 0.0)
        self.assertLess(total, 1.0)
        self.assertEqual(cfg["output"]["axes"], [0, 1, 2])

    def test_prediction_config_builds_public_options(self):
        config = load_predict_config(ROOT / "config" / "predict.yaml")
        predictor = SimpleNamespace(image_size=64, num_phases=3)
        options = config.make_options(predictor)

        self.assertTrue(str(config.run_dir))
        self.assertIsInstance(options, MPDDOptions)


if __name__ == "__main__":
    unittest.main()
