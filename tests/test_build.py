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
from src.data import AxisPatchDataset
from src.diffusion import DDPMProcess
from src.model import MPDDUNet


class BuildTest(unittest.TestCase):
    def test_axis_dataset_loads_conditions_from_configured_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {axis: root / str(axis) for axis in range(3)}
            for directory in dirs.values():
                directory.mkdir()
                Image.fromarray(
                    np.array([[0, 0], [1, 1]], dtype=np.uint8)
                ).save(directory / "phase.png")
            args = argparse.Namespace(
                data_dir=dirs,
                image_paths=None,
                crop_size=2,
                size=2,
                num_phases=2,
                segment=False,
                augment=False,
            )

            dataset = build_dataset(args)
            conditions = [dataset[index][2].item() for index in range(3)]

        self.assertIsInstance(dataset, AxisPatchDataset)
        self.assertEqual(
            dataset.paths,
            tuple(root / str(axis) / "phase.png" for axis in range(3)),
        )
        self.assertEqual(conditions, [0, 1, 2])

    def test_axis_loader_balances_conditions_despite_file_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = {axis: root / str(axis) for axis in range(3)}
            for axis, count in ((0, 1), (1, 2), (2, 5)):
                image_dir = dirs[axis]
                image_dir.mkdir()
                for index in range(count):
                    Image.fromarray(
                        np.array([[0, 0], [1, 1]], dtype=np.uint8)
                    ).save(image_dir / f"phase-{index}.png")
            args = argparse.Namespace(
                data_dir=dirs,
                image_paths=None,
                crop_size=2,
                size=2,
                num_phases=2,
                segment=False,
                augment=False,
                batch_size=30,
                num_workers=0,
            )

            dataset = build_dataset(args)
            loader = build_loader(dataset, args, device=torch.device("cpu"))
            images, fractions, conditions = next(iter(loader))

        self.assertEqual(images.shape, torch.Size([30, 1, 2, 2]))
        self.assertEqual(fractions.shape, torch.Size([30, 2]))
        self.assertEqual(torch.bincount(conditions).tolist(), [10, 10, 10])

    def test_axis_conditioned_data_rejects_image_paths(self):
        args = argparse.Namespace(
            data_dir="data/train",
            image_paths=["phase.png"],
        )

        with self.assertRaisesRegex(ValueError, "does not support image_paths"):
            build_dataset(args)

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
        self.assertTrue(hasattr(model, "anchor_encoder"))
        self.assertIsInstance(ddpm, DDPMProcess)
        self.assertEqual(ddpm.num_timesteps, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_build_model_forwards_conditioning_config(self):
        args = argparse.Namespace(
            size=8,
            num_phases=2,
            base_ch=4,
            time_dim=8,
        )

        model = build_model(args)

        self.assertTrue(hasattr(model, "anchor_encoder"))


if __name__ == "__main__":
    unittest.main()
