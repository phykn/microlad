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
from src.train import (
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
)


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
            cfg = DataConfig(
                data_dir=dirs,
                crop_size=2,
                size=2,
                num_phases=2,
                segment=False,
                augment=False,
                batch_size=1,
            )

            dataset = build_dataset(cfg)
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
            cfg = DataConfig(
                data_dir=dirs,
                crop_size=2,
                size=2,
                num_phases=2,
                segment=False,
                augment=False,
                batch_size=30,
                num_workers=0,
            )

            dataset = build_dataset(cfg)
            loader = build_loader(dataset, cfg, device=torch.device("cpu"))
            images, fractions, conditions = next(iter(loader))

        self.assertEqual(images.shape, torch.Size([30, 1, 2, 2]))
        self.assertEqual(fractions.shape, torch.Size([30, 2]))
        self.assertEqual(torch.bincount(conditions).tolist(), [10, 10, 10])

    def test_builds_mpdd_schedule_optimizer_and_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                data=DataConfig({}, 8, 8, 2, 1),
                model=ModelConfig(base_ch=4, time_dim=8),
                diffusion=DiffusionConfig(2, 0.01, 0.02),
                optimization=OptimizationConfig(1e-3, 0.01, 1.0),
                training=TrainingConfig(
                    steps=1,
                    save_every=1,
                    ema_decay=0.999,
                    frac_dropout=0.1,
                    anchor_weight=0.25,
                ),
                output=OutputConfig(tmp),
            )
            model = build_model(cfg)
            optimizer = build_optimizer(model, cfg.optimization)
            loader = [
                (
                    torch.zeros((1, 1, 8, 8)),
                    torch.tensor([[1.0, 0.0]]),
                    torch.tensor([0]),
                )
            ]
            trainer = build_trainer(
                model,
                loader,
                optimizer,
                cfg,
                torch.device("cpu"),
            )

            ddpm = build_diffusion(cfg.diffusion, device=torch.device("cpu"))
            trainer.close()

        self.assertIsInstance(model, MPDDUNet)
        self.assertTrue(hasattr(model, "anchor_encoder"))
        self.assertIsInstance(ddpm, DDPMProcess)
        self.assertEqual(ddpm.num_timesteps, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_build_model_forwards_conditioning_config(self):
        args = {
            "size": 8,
            "num_phases": 2,
            "base_ch": 4,
            "time_dim": 8,
        }

        model = build_model(args)

        self.assertTrue(hasattr(model, "anchor_encoder"))


if __name__ == "__main__":
    unittest.main()
