import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

import run_train as script


ROOT = Path(__file__).resolve().parents[1]


class RunTrainTest(unittest.TestCase):
    def test_parse_args_loads_structured_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp))
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                cfg = script.parse_args([])

        self.assertEqual(
            cfg.data.data_dir,
            {
                axis: (Path(tmp) / "train" / str(axis)).resolve()
                for axis in range(3)
            },
        )
        self.assertEqual(cfg.data.size, 64)
        self.assertEqual(cfg.training.steps, 200000)

    def test_parse_args_resolves_data_dir_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp))
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                cfg = script.parse_args([])

        self.assertEqual(
            cfg.data.data_dir,
            {
                axis: (Path(tmp) / "train" / str(axis)).resolve()
                for axis in range(3)
            },
        )

    def test_parse_args_resolves_checkpoint_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp), ckpt="weights/model.pt")
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                cfg = script.parse_args([])

        self.assertEqual(
            cfg.training.ckpt,
            (Path(tmp) / "weights" / "model.pt").resolve(),
        )

    def test_parse_args_rejects_data_dir_and_image_paths_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(
                Path(tmp),
                data_dir="train",
                image_paths=["phase.png"],
            )
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                with self.assertRaises(SystemExit):
                    script.parse_args([])

    def test_parse_args_rejects_axis_conditioned_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp), image_paths=["phase.png"])
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                with self.assertRaises(SystemExit):
                    script.parse_args([])

    def test_parse_args_rejects_single_directory_for_axis_conditioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp), data_dir="train")
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                with self.assertRaises(SystemExit):
                    script.parse_args([])

    def test_main_trains_saves_config_and_cleans_up(self):
        cfg = script.load_train_config(ROOT / "config" / "model.yaml")
        trainer = Mock()
        trainer.run_dir = Path("run/test")
        with (
            patch.object(script, "parse_args", return_value=cfg),
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
        save_config.assert_called_once_with(
            trainer.run_dir,
            cfg.as_dict(),
            name="model",
        )
        cleanup.assert_called_once_with(False)

    @staticmethod
    def _write_config(
        root: Path,
        *,
        data_dir=None,
        image_paths=None,
        ckpt=None,
    ) -> Path:
        vals = yaml.safe_load(
            (ROOT / "config" / "model.yaml").read_text(encoding="utf-8")
        )
        vals["data"]["data_dir"] = (
            {axis: f"train/{axis}" for axis in range(3)}
            if data_dir is None
            else data_dir
        )
        if image_paths is not None:
            vals["data"]["image_paths"] = image_paths
        vals["training"]["ckpt"] = ckpt
        path = root / "model.yaml"
        path.write_text(yaml.safe_dump(vals, sort_keys=False), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
