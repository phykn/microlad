import argparse
import json
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
from src.data import AxisPatchDataset, PatchDataset
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
                num_workers=0,
            )

            dataset = build_dataset(args)
            loader = build_loader(dataset, args, device=torch.device("cpu"))
            images, fractions = next(iter(loader))

        self.assertIsInstance(dataset, PatchDataset)
        self.assertEqual(loader.num_workers, 0)
        self.assertEqual(images.shape, torch.Size([2, 1, 2, 2]))
        self.assertTrue(torch.equal(fractions, torch.full((2, 2), 0.5)))

    def test_axis_manifest_allows_three_planes_to_share_one_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(
                np.array([[0, 0], [1, 1]], dtype=np.uint8)
            ).save(root / "phase.png")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "volume_axes": "ZYX",
                        "axis_sources": {"xy": ".", "xz": ".", "yz": "."},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                axis_manifest=manifest,
                axis_sampling="balanced",
                data_dir=None,
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
        self.assertEqual(dataset.paths, (root / "phase.png",) * 3)
        self.assertEqual(conditions, [0, 1, 2])

    def test_axis_loader_balances_conditions_despite_file_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for plane, count in (("xy", 1), ("xz", 2), ("yz", 5)):
                image_dir = root / plane
                image_dir.mkdir()
                for index in range(count):
                    Image.fromarray(
                        np.array([[0, 0], [1, 1]], dtype=np.uint8)
                    ).save(image_dir / f"phase-{index}.png")
                sources[plane] = plane
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "volume_axes": "ZYX",
                        "axis_sources": sources,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                axis_manifest=manifest,
                axis_sampling="balanced",
                data_dir=None,
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

    def test_axis_manifest_is_mutually_exclusive_with_other_sources(self):
        args = argparse.Namespace(
            axis_manifest="manifest.json",
            data_dir="data/train",
            image_paths=None,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_dataset(args)

    def test_rejects_removed_axis_data_contract(self):
        args = argparse.Namespace(axis_data=[])

        with self.assertRaisesRegex(ValueError, "no longer supported"):
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
                anchor_phase_loss_weight=1e-3,
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
        self.assertEqual(model.num_axis_conditions, 0)
        self.assertFalse(model.anchor_conditioning)
        self.assertEqual(model.anchor_release_step, 0)
        self.assertIsInstance(ddpm, DDPMProcess)
        self.assertEqual(ddpm.num_timesteps, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(trainer.loss.anchor_phase_loss_weight, 1e-3)

    def test_build_model_forwards_conditioning_config(self):
        args = argparse.Namespace(
            size=8,
            num_phases=2,
            num_axis_conditions=3,
            anchor_conditioning=True,
            anchor_release_step=3,
            base_ch=4,
            time_dim=8,
        )

        model = build_model(args)

        self.assertEqual(model.num_axis_conditions, 3)
        self.assertTrue(model.anchor_conditioning)
        self.assertEqual(model.anchor_release_step, 3)


if __name__ == "__main__":
    unittest.main()
