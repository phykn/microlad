import tempfile
import unittest
from pathlib import Path

import torch

from src.pipelines.training.misc.run import save_checkpoint, setup_run_dirs


class RunTest(unittest.TestCase):
    def test_setup_skips_writer_and_directories_on_non_main_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, log_dir, weight_dir, last_weight_dir, writer = setup_run_dirs(
                run_root=tmp,
                component="vae",
                is_main_process=False,
            )

            self.assertIsNone(writer)
            self.assertFalse(log_dir.exists())
            self.assertFalse(last_weight_dir.exists())
            self.assertEqual(weight_dir.parent, run_dir / "weight")

    def test_checkpoint_skips_non_main_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=0,
                save_every=2,
                weight_dir=root / "weight",
                last_weight_dir=root / "weight" / "last",
                is_main_process=False,
            )

            self.assertFalse((root / "weight").exists())

    def test_checkpoint_writes_initial_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=0,
                save_every=2,
                weight_dir=root / "weight",
                last_weight_dir=root / "weight" / "last",
                is_main_process=True,
            )

            for path in (
                root / "weight" / "0" / "model.pt",
                root / "weight" / "last" / "model.pt",
            ):
                self.assertEqual(torch.load(path, weights_only=True)["step"], 0)

    def test_checkpoint_writes_interval_and_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=2,
                save_every=2,
                weight_dir=root / "weight",
                last_weight_dir=root / "weight" / "last",
                is_main_process=True,
            )

            for path in (
                root / "weight" / "2" / "model.pt",
                root / "weight" / "last" / "model.pt",
            ):
                self.assertEqual(torch.load(path, weights_only=True)["step"], 2)

    def test_checkpoint_skips_steps_outside_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1,
                save_every=2,
                weight_dir=root / "weight",
                last_weight_dir=root / "weight" / "last",
                is_main_process=True,
            )

            self.assertFalse((root / "weight").exists())
