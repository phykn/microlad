import tempfile
import unittest
from pathlib import Path

import yaml

from gen_fake import load_config, parse_args


class GenerateCriticFakesEntrypointTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        args = parse_args(["--check"])
        load_config(args.config)

    def test_rejects_invalid_count_before_model_loading(self):
        values = {
            "models": {
                "vae_run_dir": "run/vae",
                "diffusion_run_dir": "run/diffusion",
            },
            "data": {"data_dir": "data"},
            "generation": {
                "num_volumes": 0,
                "unconditional_ratio": 0.1,
                "progress": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num_volumes"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
