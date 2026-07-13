import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.app.runtime import (
    build_dataset,
    build_denoiser,
    build_loader,
    build_vae,
    build_optimizer,
    copy_vae_run,
    cleanup_distributed,
    apply_vae_defaults,
    load_predictor,
    load_predict_config,
    load_run_vae,
    load_defaults,
    save_run_config,
    setup_device,
    wrap_distributed,
)
from src.app.runtime.model import build_ddpm
from src.pipelines.data import PatchDataset
from src.modeling.diffusion import DDPMProcess, TimeUNet
from src.modeling.vae import PatchVAE
from src.app.api import PredictOptions, Predictor


def save_image(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def write_predictor_run(
    run_dir: Path,
    *,
    image_size: int = 8,
    latent_size: int = 4,
    latent_ch: int = 2,
    write_vae_checkpoint: bool = True,
    write_diffusion_checkpoint: bool = True,
    checkpoint_vae_latent_ch: int | None = None,
) -> None:
    vae_args = argparse.Namespace(
        size=image_size,
        crop_size=image_size,
        latent_size=latent_size,
        latent_ch=latent_ch,
        base_ch=4,
        max_ch=8,
        num_phases=2,
        segment=False,
    )
    diffusion_args = argparse.Namespace(
        base_ch=4,
        time_dim=8,
        timesteps=4,
        beta_start=0.01,
        beta_end=0.02,
    )

    if write_vae_checkpoint:
        checkpoint_vae_args = argparse.Namespace(
            size=image_size,
            latent_size=latent_size,
            latent_ch=checkpoint_vae_latent_ch or latent_ch,
            base_ch=4,
            max_ch=8,
            num_phases=2,
        )
        source_vae = build_vae(checkpoint_vae_args)
        vae_ckpt = run_dir / "weight" / "vae" / "last" / "model.pt"
        vae_ckpt.parent.mkdir(parents=True)
        torch.save({"model": source_vae.state_dict()}, vae_ckpt)

    if write_diffusion_checkpoint:
        source_diffusion = build_denoiser(
            argparse.Namespace(latent_ch=latent_ch, base_ch=4, time_dim=8)
        )
        diffusion_ckpt = run_dir / "weight" / "diffusion" / "last" / "model.pt"
        diffusion_ckpt.parent.mkdir(parents=True)
        torch.save({"model": source_diffusion.state_dict()}, diffusion_ckpt)

    save_run_config(run_dir, vae_args, name="vae")
    save_run_config(run_dir, diffusion_args, name="diffusion")


class BuildTest(unittest.TestCase):
    def test_load_defaults_flattens_nested_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.yaml"
            path.write_text(
                "\n".join(
                    [
                        "data:",
                        "  data_dir: data",
                        "  crop_size: 64",
                        "training:",
                        "  steps: 10",
                    ]
                ),
                encoding="utf-8",
            )

            defaults = load_defaults(path)

        self.assertEqual(
            defaults,
            {
                "data_dir": "data",
                "crop_size": 64,
                "steps": 10,
            },
        )

    def test_load_predict_config_preserves_grouped_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predict.yaml"
            path.write_text(
                "\n".join(
                    [
                        "progress: false",
                        "phase_fractions: [0.25, 0.15, 0.60]",
                        "phase_fraction_tolerance: 0.02",
                        "segment_anchors: true",
                        "joint:",
                        "  steps: 12",
                        "  batch_size: 3",
                        "  decode_batch_size: null",
                        "critic:",
                        "  steps: 7",
                        "  weight: 0.05",
                        "refine:",
                        "  candidates: [0, 1, 2]",
                        "quality:",
                        "  anchor_tolerance: 0.12",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_predict_config(path)

        self.assertEqual(config["joint"].steps, 12)
        self.assertEqual(config["joint"].batch_size, 3)
        self.assertIsNone(config["joint"].decode_batch_size)
        self.assertFalse(config["progress"])
        self.assertEqual(config["phase_fractions"], (0.25, 0.15, 0.60))
        self.assertEqual(config["phase_fraction_tolerance"], 0.02)
        self.assertTrue(config["segment_anchors"])
        self.assertEqual(config["critic"].steps, 7)
        self.assertEqual(config["critic"].weight, 0.05)
        self.assertEqual(config["refine"].candidates, (0, 1, 2))
        self.assertEqual(config["quality"].anchor_tolerance, 0.12)

    def test_build_dataset_expands_image_files_from_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(root / "b.png", np.zeros((8, 8), dtype=np.uint8))
            save_image(root / "a.png", np.ones((8, 8), dtype=np.uint8))
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            args = argparse.Namespace(
                data_dir=root,
                crop_size=8,
                size=4,
                num_phases=2,
                segment=False,
                augment=False,
            )

            dataset = build_dataset(args)

        self.assertIsInstance(dataset, PatchDataset)
        self.assertEqual(
            [path.name for path in dataset.image_paths], ["a.png", "b.png"]
        )
        self.assertEqual(dataset.crop_size, 8)
        self.assertEqual(dataset.image_size, 4)

    def test_build_dataset_accepts_single_image_path_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "phase.png"
            save_image(image_path, np.zeros((8, 8), dtype=np.uint8))
            args = argparse.Namespace(
                image_paths=str(image_path),
                crop_size=8,
                size=4,
                num_phases=2,
                segment=False,
                augment=False,
            )

            dataset = build_dataset(args)
            sample = dataset[0]

        self.assertEqual(dataset.image_paths, [image_path])
        self.assertEqual(sample.shape, torch.Size([1, 4, 4]))

    def test_build_dataset_rejects_empty_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                data_dir=Path(tmp),
                crop_size=8,
                size=4,
                num_phases=2,
                segment=False,
                augment=False,
            )

            with self.assertRaisesRegex(ValueError, "image_paths"):
                build_dataset(args)

    def test_build_loader_samples_batch_with_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_image(root / "phase.png", np.zeros((8, 8), dtype=np.uint8))
            args = argparse.Namespace(
                data_dir=root,
                crop_size=8,
                size=4,
                num_phases=2,
                segment=False,
                augment=False,
                batch_size=8,
            )
            dataset = build_dataset(args)
            loader = build_loader(
                dataset,
                args,
                device=torch.device("cpu"),
            )

            batch = next(loader)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(batch.shape, torch.Size([8, 1, 4, 4]))

    def test_build_vae_uses_model_config(self):
        args = argparse.Namespace(
            size=64,
            latent_size=16,
            latent_ch=2,
            num_phases=5,
            base_ch=8,
            max_ch=16,
        )

        vae = build_vae(args)

        self.assertIsInstance(vae, PatchVAE)
        self.assertEqual(vae.image_size, 64)
        self.assertEqual(vae.latent_size, 16)
        self.assertEqual(vae.latent_ch, 2)
        self.assertEqual(vae.num_phases, 5)
        self.assertEqual(vae.channels, (8, 16, 16))

    def test_build_denoiser_uses_model_config(self):
        args = argparse.Namespace(latent_ch=4, base_ch=8, time_dim=16)

        model = build_denoiser(args)

        self.assertIsInstance(model, TimeUNet)
        self.assertEqual(model.latent_ch, 4)
        self.assertEqual(model.base_ch, 8)
        self.assertEqual(model.time_dim, 16)

    def test_save_run_config_writes_flat_config_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = argparse.Namespace(num_phases=3, vae_ckpt="vae.pt")

            save_run_config(run_dir, args, name="vae")

            defaults = load_defaults(run_dir / "vae.yaml")

        self.assertEqual(defaults["num_phases"], 3)
        self.assertEqual(defaults["vae_ckpt"], "vae.pt")

    def test_load_run_vae_uses_vae_config_and_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = argparse.Namespace(
                size=64,
                latent_size=16,
                latent_ch=2,
                base_ch=8,
                max_ch=16,
                num_phases=2,
            )
            source = build_vae(args)
            with torch.no_grad():
                for parameter in source.parameters():
                    parameter.fill_(0.5)
            checkpoint = run_dir / "weight" / "vae" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": source.state_dict()}, checkpoint)
            save_run_config(run_dir, args, name="vae")

            vae = load_run_vae(run_dir, device=torch.device("cpu"))

        self.assertIsInstance(vae, PatchVAE)
        self.assertFalse(vae.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in vae.parameters())
        )

    def test_load_predictor_uses_run_dir_for_notebook_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)

            predictor = load_predictor(run_dir, device="cpu")
            with self.assertWarnsRegex(RuntimeWarning, "least-violation"):
                volume, stats = predictor.predict(PredictOptions(num_phases=2))

        self.assertIsInstance(predictor, Predictor)
        self.assertEqual(predictor.device, torch.device("cpu"))
        self.assertEqual(predictor.ddpm.num_timesteps, 4)
        self.assertEqual(volume.shape, torch.Size([8, 8, 8]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_load_predictor_reports_missing_diffusion_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "diffusion config"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_missing_diffusion_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir, write_diffusion_checkpoint=False)

            with self.assertRaisesRegex(FileNotFoundError, "diffusion checkpoint"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_incompatible_vae_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir, checkpoint_vae_latent_ch=3)

            with self.assertRaisesRegex(ValueError, "vae checkpoint"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_incomplete_vae_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)
            (run_dir / "vae.yaml").write_text(
                "\n".join(
                    [
                        "size: 8",
                        "crop_size: 8",
                        "latent_size: 4",
                        "latent_ch: 2",
                        "base_ch: 4",
                        "max_ch: 8",
                        "segment: false",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "vae config.*num_phases"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_incomplete_diffusion_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)
            (run_dir / "diffusion.yaml").write_text(
                "\n".join(
                    [
                        "base_ch: 4",
                        "time_dim: 8",
                        "timesteps: 4",
                        "beta_end: 0.02",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "diffusion config.*beta_start"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_malformed_vae_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)
            (run_dir / "vae.yaml").write_text("vae: [\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vae config.*malformed"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_malformed_diffusion_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)
            (run_dir / "diffusion.yaml").write_text(
                "diffusion: [\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "diffusion config.*malformed"):
                load_predictor(run_dir, device="cpu")

    def test_load_predictor_reports_corrupt_diffusion_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_predictor_run(run_dir)
            (run_dir / "weight" / "diffusion" / "last" / "model.pt").write_text(
                "not a checkpoint",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "diffusion checkpoint"):
                load_predictor(run_dir, device="cpu")

    def test_apply_vae_defaults_uses_vae_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            save_run_config(
                run_dir,
                argparse.Namespace(
                    crop_size=128,
                    size=64,
                    segment=True,
                    latent_ch=2,
                    num_phases=3,
                ),
                name="vae",
            )
            args = argparse.Namespace(vae_run_dir=run_dir)

            filled = apply_vae_defaults(args)

        self.assertIs(filled, args)
        self.assertEqual(args.crop_size, 128)
        self.assertEqual(args.size, 64)
        self.assertTrue(args.segment)
        self.assertEqual(args.latent_ch, 2)
        self.assertEqual(args.num_phases, 3)

    def test_apply_vae_defaults_rejects_incompatible_latent_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            save_run_config(
                run_dir,
                argparse.Namespace(
                    size=40,
                    crop_size=80,
                    segment=False,
                    latent_size=10,
                    latent_ch=2,
                    num_phases=3,
                ),
                name="vae",
            )
            args = argparse.Namespace(vae_run_dir=run_dir)

            with self.assertRaisesRegex(ValueError, "latent_size"):
                apply_vae_defaults(args)

    def test_copy_vae_run_copies_config_and_last_weight_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "vae-run"
            target = Path(tmp) / "diffusion-run"
            save_run_config(
                source,
                argparse.Namespace(
                    size=64,
                    crop_size=128,
                    segment=False,
                    latent_ch=2,
                    num_phases=3,
                ),
                name="vae",
            )
            weight = source / "weight" / "vae" / "last" / "model.pt"
            weight.parent.mkdir(parents=True)
            torch.save({"model": {}}, weight)
            extra = source / "weight" / "vae" / "1" / "model.pt"
            extra.parent.mkdir(parents=True)
            torch.save({"model": {}}, extra)

            copy_vae_run(source, target)

            self.assertTrue((target / "vae.yaml").is_file())
            self.assertTrue((target / "weight" / "vae" / "last" / "model.pt").is_file())
            self.assertFalse((target / "weight" / "vae" / "1").exists())

    def test_build_ddpm_uses_diffusion_config(self):
        args = argparse.Namespace(timesteps=8, beta_start=0.01, beta_end=0.02)

        ddpm = build_ddpm(args, device=torch.device("cpu"))

        self.assertIsInstance(ddpm, DDPMProcess)
        self.assertEqual(ddpm.num_timesteps, 8)
        self.assertTrue(torch.allclose(ddpm.betas[0], torch.tensor(0.01)))
        self.assertTrue(torch.allclose(ddpm.betas[-1], torch.tensor(0.02)))

    def test_build_optimizer_uses_adamw_and_lr(self):
        model = torch.nn.Linear(1, 1)
        args = argparse.Namespace(lr=1e-3, weight_decay=0.01)

        optimizer = build_optimizer(model, args)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_setup_device_uses_plain_device_without_distributed_env(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "src.app.runtime.distributed.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "src.app.runtime.distributed.dist.init_process_group"
            ) as init_process_group,
        ):
            device, local_rank, distributed = setup_device()

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(local_rank, 0)
        self.assertFalse(distributed)
        init_process_group.assert_not_called()

    def test_setup_device_initializes_cpu_distributed_from_env(self):
        with (
            patch.dict(
                os.environ,
                {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"},
                clear=True,
            ),
            patch(
                "src.app.runtime.distributed.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "src.app.runtime.distributed.dist.init_process_group"
            ) as init_process_group,
        ):
            device, local_rank, distributed = setup_device()

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(local_rank, 1)
        self.assertTrue(distributed)
        init_process_group.assert_called_once_with(backend="gloo")

    def test_wrap_distributed_uses_ddp_without_cpu_device_ids(self):
        model = torch.nn.Linear(1, 1)

        with patch("src.app.runtime.distributed.DistributedDataParallel") as ddp:
            ddp.return_value = "wrapped"
            wrapped = wrap_distributed(model, local_rank=1, distributed=True)

        self.assertEqual(wrapped, "wrapped")
        ddp.assert_called_once_with(model)

    def test_cleanup_distributed_destroys_initialized_process_group(self):
        with (
            patch("src.app.runtime.distributed.dist.is_available", return_value=True),
            patch("src.app.runtime.distributed.dist.is_initialized", return_value=True),
            patch("src.app.runtime.distributed.dist.destroy_process_group") as destroy,
        ):
            cleanup_distributed(enabled=True)

        destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
