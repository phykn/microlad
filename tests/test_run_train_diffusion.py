import argparse
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.build import save_run_config
from src.models import PatchVAE


def load_script():
    try:
        return importlib.import_module("run_train_diffusion")
    except ModuleNotFoundError as exc:
        raise AssertionError("run_train_diffusion.py should exist") from exc


def write_image(path: Path) -> None:
    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[:, 32:] = 1
    Image.fromarray(pixels).save(path)


def write_config(path: Path, data_dir: Path, vae_run_dir: Path, run_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  data_dir: {data_dir.as_posix()}",
                "  crop_size: 64",
                "  segment: false",
                "  augment: false",
                "  batch_size: 1",
                "model:",
                "  base_ch: 4",
                "  time_dim: 8",
                "diffusion:",
                "  timesteps: 4",
                "  beta_start: 0.0001",
                "  beta_end: 0.02",
                "optimization:",
                "  lr: 0.0001",
                "  weight_decay: 0.0",
                "  clip_grad_norm: 1.0",
                "training:",
                "  steps: 1",
                "  save_every: 1",
                "output:",
                f"  vae_run_dir: {vae_run_dir.as_posix()}",
                f"  run_root: {run_root.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )


def write_vae_run(run_dir: Path) -> None:
    vae_args = argparse.Namespace(
        image_size=64,
        latent_size=16,
        latent_ch=2,
        base_ch=4,
        max_ch=8,
        num_phases=2,
    )
    vae = PatchVAE(
        image_size=vae_args.image_size,
        latent_size=vae_args.latent_size,
        latent_ch=vae_args.latent_ch,
        base_ch=vae_args.base_ch,
        max_ch=vae_args.max_ch,
    )
    checkpoint = run_dir / "weight" / "vae" / "last" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model": vae.state_dict()}, checkpoint)
    save_run_config(run_dir, vae_args, name="vae")


class RunTrainDiffusionTest(unittest.TestCase):
    def test_parse_args_loads_diffusion_config(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "diffusion.yaml"
            data_dir = root / "data"
            vae_run_dir = root / "vae-run"
            run_root = root / "run"
            write_vae_run(vae_run_dir)
            write_config(config, data_dir, vae_run_dir, run_root)
            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)

            args = script.parse_args_from_list([])

        self.assertEqual(args.data_dir, data_dir.as_posix())
        self.assertEqual(args.vae_run_dir, vae_run_dir.as_posix())
        self.assertEqual(args.run_root, run_root.as_posix())
        self.assertEqual(args.size, 64)
        self.assertEqual(args.num_phases, 2)
        self.assertEqual(args.latent_ch, 2)
        self.assertEqual(args.timesteps, 4)

    def test_main_trains_one_step_and_writes_checkpoint(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            write_image(data_dir / "phase.png")

            vae_run_dir = root / "vae-run"
            run_root = root / "run"
            write_vae_run(vae_run_dir)
            config = root / "diffusion.yaml"
            write_config(config, data_dir, vae_run_dir, run_root)

            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)
            with patch.object(sys, "argv", ["run_train_diffusion.py"]):
                script.main()

            run_dirs = list(run_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "vae.yaml").is_file())
            self.assertTrue((run_dir / "weight" / "vae" / "last" / "model.pt").is_file())
            self.assertTrue((run_dir / "diffusion.yaml").is_file())
            self.assertTrue(
                (run_dir / "weight" / "diffusion" / "last" / "model.pt").is_file()
            )
            self.assertFalse((vae_run_dir / "weight" / "diffusion").exists())


if __name__ == "__main__":
    unittest.main()
