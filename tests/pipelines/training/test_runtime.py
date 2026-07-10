import tempfile
import unittest
from pathlib import Path

import torch

from src.pipelines.training.runtime import next_batch, save_checkpoint, setup_run_dirs


class SaveCheckpointTest(unittest.TestCase):
    def test_next_batch_restarts_finite_reiterable_dataloader(self):
        dataloader = [torch.tensor([1]), torch.tensor([2])]
        iterator = iter(dataloader)

        first, iterator = next_batch(dataloader, iterator)
        second, iterator = next_batch(dataloader, iterator)
        third, iterator = next_batch(dataloader, iterator)

        self.assertTrue(torch.equal(first, torch.tensor([1])))
        self.assertTrue(torch.equal(second, torch.tensor([2])))
        self.assertTrue(torch.equal(third, torch.tensor([1])))

    def test_next_batch_rejects_exhausted_non_reiterable_iterator(self):
        dataloader = iter([torch.tensor([1])])
        iterator = iter(dataloader)

        first, iterator = next_batch(dataloader, iterator)

        self.assertTrue(torch.equal(first, torch.tensor([1])))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            next_batch(dataloader, iterator)

    def test_setup_run_dirs_skips_writer_and_directories_on_non_main_process(self):
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

    def test_save_checkpoint_skips_non_main_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=1,
                save_every=1,
                weight_dir=root / "weight",
                last_weight_dir=root / "weight" / "last",
                is_main_process=False,
            )

            self.assertFalse((root / "weight").exists())

    def test_save_checkpoint_writes_latest_and_scheduled_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_dir = root / "weight"
            last_weight_dir = weight_dir / "last"

            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=2,
                save_every=2,
                weight_dir=weight_dir,
                last_weight_dir=last_weight_dir,
                is_main_process=True,
            )

            for path in (weight_dir / "2" / "model.pt", last_weight_dir / "model.pt"):
                self.assertEqual(torch.load(path, weights_only=True)["step"], 2)
