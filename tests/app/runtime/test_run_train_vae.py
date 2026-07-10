import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import run_train_vae as script
from src.app.runtime import load_run_vae


def write_image(path: Path) -> None:
    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[:, 32:] = 1
    Image.fromarray(pixels).save(path)


def write_config(path: Path, data_dir: Path, run_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  data_dir: {data_dir.as_posix()}",
                "  crop_size: 64",
                "  size: 64",
                "  num_phases: 2",
                "  segment: false",
                "  augment: false",
                "  batch_size: 1",
                "model:",
                "  latent_size: 16",
                "  latent_ch: 2",
                "  base_ch: 4",
                "  max_ch: 8",
                "loss:",
                "  beta: 1.0",
                "optimization:",
                "  lr: 0.0001",
                "  weight_decay: 0.0",
                "  clip_grad_norm: 1.0",
                "training:",
                "  steps: 1",
                "  save_every: 1",
                "output:",
                f"  run_root: {run_root.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )


class FailingTrainer:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.closed = False

    def train(self):
        raise RuntimeError("training failed")

    def close(self) -> None:
        self.closed = True


class CloseFailTrainer:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def train(self):
        return None

    def close(self) -> None:
        raise RuntimeError("close failed")


class DeviceModel:
    def to(self, device):
        return self


class RunTrainVAETest(unittest.TestCase):
    def test_parse_args_loads_vae_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            run_root = root / "run"
            config = root / "vae.yaml"
            write_config(config, data_dir, run_root)
            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)

            args = script.parse_args_from_list([])

        self.assertEqual(args.data_dir, data_dir.as_posix())
        self.assertEqual(args.run_root, run_root.as_posix())
        self.assertEqual(args.size, 64)
        self.assertEqual(args.num_phases, 2)

    def test_main_trains_one_step_and_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            write_image(data_dir / "phase.png")
            run_root = root / "run"
            config = root / "vae.yaml"
            write_config(config, data_dir, run_root)

            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)
            with patch.object(sys, "argv", ["run_train_vae.py"]):
                script.main()

            run_dirs = list(run_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "vae.yaml").is_file())
            self.assertTrue((run_dir / "log" / "vae").is_dir())
            self.assertTrue(
                list((run_dir / "log" / "vae").glob("events.out.tfevents.*"))
            )
            self.assertTrue((run_dir / "weight" / "vae" / "1" / "model.pt").is_file())
            checkpoint_path = run_dir / "weight" / "vae" / "last" / "model.pt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["step"], 1)
            self.assertIn("model", checkpoint)
            self.assertIn("optimizer", checkpoint)
            vae = load_run_vae(run_dir, torch.device("cpu"))

        self.assertFalse(vae.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in vae.parameters())
        )

    def test_main_closes_trainer_when_training_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            write_image(data_dir / "phase.png")
            run_root = root / "run"
            config = root / "vae.yaml"
            write_config(config, data_dir, run_root)
            trainer = FailingTrainer(root / "failed-run")

            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)

            with (
                patch.object(sys, "argv", ["run_train_vae.py"]),
                patch.object(script, "setup_device", return_value=("cpu", 0, False)),
                patch.object(script, "build_dataset", return_value=object()),
                patch.object(script, "build_loader", return_value=iter([object()])),
                patch.object(script, "build_vae", return_value=DeviceModel()),
                patch.object(script, "build_optimizer", return_value=object()),
                patch.object(script, "build_vae_trainer", return_value=trainer),
                patch.object(script, "save_run_config"),
                patch.object(script, "cleanup_distributed") as cleanup,
                self.assertRaisesRegex(RuntimeError, "training failed"),
            ):
                script.main()

        self.assertTrue(trainer.closed)
        cleanup.assert_called_once_with(False)

    def test_main_cleans_up_distributed_when_close_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            write_image(data_dir / "phase.png")
            run_root = root / "run"
            config = root / "vae.yaml"
            write_config(config, data_dir, run_root)
            trainer = CloseFailTrainer(root / "failed-close-run")

            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)

            with (
                patch.object(sys, "argv", ["run_train_vae.py"]),
                patch.object(script, "setup_device", return_value=("cpu", 0, False)),
                patch.object(script, "build_dataset", return_value=object()),
                patch.object(script, "build_loader", return_value=iter([object()])),
                patch.object(script, "build_vae", return_value=DeviceModel()),
                patch.object(script, "build_optimizer", return_value=object()),
                patch.object(script, "build_vae_trainer", return_value=trainer),
                patch.object(script, "save_run_config"),
                patch.object(script, "cleanup_distributed") as cleanup,
                self.assertRaisesRegex(RuntimeError, "close failed"),
            ):
                script.main()

        cleanup.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
