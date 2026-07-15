import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import run_train as script


class RunTrainTest(unittest.TestCase):
    def test_parse_args_loads_flattened_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "model.yaml"
            config.write_text(
                "data:\n  data_dir: data/train\n  size: 8\ntraining:\n  steps: 2\n",
                encoding="utf-8",
            )
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                args = script.parse_args([])

        self.assertEqual(args.data_dir, "data/train")
        self.assertEqual(args.size, 8)
        self.assertEqual(args.steps, 2)

    def test_main_trains_saves_config_and_cleans_up(self):
        args = argparse.Namespace(steps=2)
        trainer = Mock()
        trainer.run_dir = Path("run/test")
        with (
            patch.object(script, "parse_args", return_value=args),
            patch.object(
                script.distributed,
                "setup",
                return_value=(torch.device("cpu"), 0, False),
            ),
            patch.object(script, "build_dataset", return_value=object()),
            patch.object(script, "build_loader", return_value=object()),
            patch.object(script, "build_model", return_value=torch.nn.Linear(1, 1)),
            patch.object(
                script.distributed,
                "wrap",
                side_effect=lambda model, **_: model,
            ),
            patch.object(script, "build_optimizer", return_value=object()),
            patch.object(script, "build_trainer", return_value=trainer),
            patch.object(script, "save_config") as save_config,
            patch.object(script.distributed, "cleanup") as cleanup,
        ):
            script.main()

        trainer.train.assert_called_once_with()
        trainer.close.assert_called_once_with()
        save_config.assert_called_once_with(trainer.run_dir, args, name="model")
        cleanup.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
