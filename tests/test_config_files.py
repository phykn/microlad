from pathlib import Path
from types import SimpleNamespace
import unittest

from src.misc import load_config, load_mapping
from src.predict import MPDDOptions, load_predict_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigFileTest(unittest.TestCase):
    def test_training_config_has_runtime_inputs(self):
        config = load_config(ROOT / "config" / "model.yaml")
        required = {
            "axis_manifest",
            "axis_sampling",
            "crop_size",
            "size",
            "num_phases",
            "num_axis_conditions",
            "base_ch",
            "time_dim",
            "timesteps",
            "beta_start",
            "beta_end",
            "lr",
            "anchor_phase_loss_weight",
            "steps",
            "run_root",
        }

        self.assertFalse(required - config.keys())
        self.assertEqual(config["num_axis_conditions"], 3)
        self.assertEqual(config["axis_sampling"], "balanced")
        self.assertEqual(config["anchor_phase_loss_weight"], 0.0)
        self.assertEqual(
            config["axis_manifest"],
            "../data/generated/manifest.json",
        )

    def test_simulation_config_uses_random_hard_spheres_and_all_planes(self):
        config = load_mapping(ROOT / "config" / "simul.yaml")

        self.assertEqual(config["geometry"]["mode"], "dry")
        self.assertNotIn("seed", config["geometry"])
        self.assertEqual(config["geometry"]["shape"], "sphere")
        self.assertEqual(config["geometry"]["alignment_axis"], "z")
        self.assertGreater(config["geometry"]["elongation"], 1.0)
        self.assertLess(
            config["geometry"]["big_fraction"]
            + config["geometry"]["small_fraction"],
            0.5,
        )
        self.assertEqual(config["export"]["planes"], ["xy", "xz", "yz"])

    def test_prediction_config_builds_public_options(self):
        config = load_predict_config(ROOT / "config" / "predict.yaml")
        predictor = SimpleNamespace(image_size=64, num_phases=3)
        options = config.make_options(predictor)

        self.assertTrue(str(config.run_dir))
        self.assertIsInstance(options, MPDDOptions)


if __name__ == "__main__":
    unittest.main()
