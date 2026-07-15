import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.build import (
    build_dataset,
    build_diffusion,
    build_loader,
    build_model,
    build_optimizer,
    build_trainer,
)
from src.data import PatchDataset
from src.diffusion import DDPMProcess
from src.model import MPDDUNet


class BuildTest(unittest.TestCase):
    def test_build_dataset_and_loader_return_images_with_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.array([[0, 0], [1, 1]], dtype=np.uint8)).save(
                root / "phase.png"
            )
            args = argparse.Namespace(
                data_dir=root,
                image_paths=None,
                crop_size=2,
                size=2,
                num_phases=2,
                segment=False,
                augment=False,
                batch_size=2,
            )

            dataset = build_dataset(args)
            images, fractions = next(
                build_loader(dataset, args, device=torch.device("cpu"))
            )

        self.assertIsInstance(dataset, PatchDataset)
        self.assertEqual(images.shape, torch.Size([2, 1, 2, 2]))
        self.assertTrue(torch.equal(fractions, torch.full((2, 2), 0.5)))

    def test_builds_mpdd_schedule_optimizer_and_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                size=8,
                num_phases=2,
                base_ch=4,
                time_dim=8,
                timesteps=2,
                beta_start=0.01,
                beta_end=0.02,
                lr=1e-3,
                weight_decay=0.01,
                steps=1,
                run_root=tmp,
                save_every=1,
                clip_grad_norm=1.0,
                ema_decay=0.999,
                condition_dropout=0.1,
                warmup_steps=0,
            )
            model = build_model(args)
            optimizer = build_optimizer(model, args)
            loader = [
                (
                    torch.zeros((1, 1, 8, 8)),
                    torch.tensor([[1.0, 0.0]]),
                )
            ]
            trainer = build_trainer(
                model,
                loader,
                optimizer,
                args,
                torch.device("cpu"),
            )

            ddpm = build_diffusion(args, device=torch.device("cpu"))
            trainer.close()

        self.assertIsInstance(model, MPDDUNet)
        self.assertIsInstance(ddpm, DDPMProcess)
        self.assertEqual(ddpm.num_timesteps, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)


if __name__ == "__main__":
    unittest.main()
