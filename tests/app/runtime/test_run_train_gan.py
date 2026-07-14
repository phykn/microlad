import argparse
import tempfile
import unittest
from pathlib import Path

import torch

import run_train_gan as script
from src.app.runtime import (
    build_critic,
    build_generator,
    load_run_critic,
    load_run_generator,
    load_predictor,
    save_run_config,
)
from src.modeling.diffusion import TimeUNet
from src.modeling.vae import PatchVAE


def write_vae_run(run_dir: Path) -> None:
    vae_args = argparse.Namespace(
        size=64,
        crop_size=64,
        latent_size=16,
        latent_ch=2,
        base_ch=4,
        max_ch=8,
        num_phases=2,
        segment=False,
    )
    vae = PatchVAE(
        image_size=64,
        latent_size=16,
        latent_ch=2,
        num_phases=2,
        base_ch=4,
        max_ch=8,
    )
    checkpoint = run_dir / "weight" / "vae" / "last" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": vae.state_dict()}, checkpoint)
    save_run_config(run_dir, vae_args, name="vae")


def write_diffusion_run(run_dir: Path) -> None:
    args = argparse.Namespace(
        base_ch=4,
        time_dim=8,
        timesteps=4,
        beta_start=0.001,
        beta_end=0.02,
    )
    denoiser = TimeUNet(latent_ch=2, base_ch=4, time_dim=8)
    checkpoint = run_dir / "weight" / "diffusion" / "last" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": denoiser.state_dict()}, checkpoint)
    save_run_config(run_dir, args, name="diffusion")


def write_config(path: Path, source: Path, run_root: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "data:",
                "  data_dir: data",
                "  augment: true",
                "  batch_size: 2",
                "model:",
                "  noise_ch: 8",
                "  generator_ch: 8",
                "  critic_ch: 4",
                "loss:",
                "  gp_weight: 10.0",
                "optimization:",
                "  generator_lr: 0.0001",
                "  critic_lr: 0.0001",
                "  betas: [0.0, 0.9]",
                "  clip_grad_norm: 1.0",
                "training:",
                "  steps: 1",
                "  critic_steps: 1",
                "  save_every: 1",
                "output:",
                f"  vae_run_dir: {source.as_posix()}",
                f"  run_root: {run_root.as_posix()}",
            )
        ),
        encoding="utf-8",
    )


class RunTrainGANTest(unittest.TestCase):
    def test_config_inherits_categorical_vae_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            config = root / "gan.yaml"
            write_vae_run(source)
            write_config(config, source, root / "run")
            old_config = script.DEFAULT_CONFIG
            script.DEFAULT_CONFIG = str(config)
            self.addCleanup(setattr, script, "DEFAULT_CONFIG", old_config)

            args = script.parse_args_from_list([])

        self.assertEqual(args.size, 64)
        self.assertEqual(args.latent_size, 16)
        self.assertEqual(args.latent_ch, 2)
        self.assertEqual(args.num_phases, 2)
        self.assertFalse(args.segment)

    def test_saved_gan_models_load_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae_run = root / "vae"
            diffusion_run = root / "diffusion"
            gan_run = root / "gan"
            write_vae_run(vae_run)
            write_vae_run(gan_run)
            write_diffusion_run(diffusion_run)
            args = argparse.Namespace(
                latent_ch=2,
                latent_size=16,
                num_phases=2,
                noise_ch=8,
                generator_ch=8,
                critic_ch=4,
            )
            generator = build_generator(args)
            critic = build_critic(args)
            checkpoint = gan_run / "weight" / "gan" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "critic": critic.state_dict(),
                },
                checkpoint,
            )
            save_run_config(gan_run, args, name="gan")

            loaded_generator = load_run_generator(gan_run, torch.device("cpu"))
            loaded_critic = load_run_critic(gan_run, torch.device("cpu"))
            predictor = load_predictor(
                vae_run,
                diffusion_run,
                gan_run,
                device="cpu",
            )

        self.assertFalse(loaded_generator.training)
        self.assertFalse(loaded_critic.training)
        self.assertIsNotNone(predictor.critic)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in loaded_critic.parameters())
        )


if __name__ == "__main__":
    unittest.main()
