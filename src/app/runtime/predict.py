import argparse
from pathlib import Path

import torch

from src.app.api import Predictor
from src.app.runtime.load import load_denoiser, load_run_critic, load_run_vae
from src.app.runtime.model import build_ddpm
from src.app.runtime.run import (
    apply_vae_defaults,
    checkpoint_path,
    load_run_config,
    require_values,
)


def load_predictor(
    vae_run_dir: str | Path,
    diffusion_run_dir: str | Path,
    gan_run_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> Predictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    diffusion_config = load_run_config(diffusion_run_dir, "diffusion")
    require_values(
        diffusion_config,
        "diffusion config",
        "base_ch",
        "time_dim",
        "timesteps",
        "beta_start",
        "beta_end",
    )
    args = argparse.Namespace(**diffusion_config)
    args.vae_run_dir = vae_run_dir
    apply_vae_defaults(args)
    args.diffusion_ckpt = checkpoint_path(diffusion_run_dir, "diffusion")
    critic = None
    if gan_run_dir is not None:
        vae_config = load_run_config(vae_run_dir, "vae")
        gan_config = load_run_config(gan_run_dir, "gan")
        require_values(
            gan_config,
            "gan config",
            "latent_ch",
            "latent_size",
            "num_phases",
        )
        for name in ("latent_ch", "latent_size", "num_phases"):
            if gan_config[name] != vae_config[name]:
                raise ValueError(f"GAN {name} must match the selected VAE run.")
        critic = load_run_critic(gan_run_dir, device=device)
    return Predictor(
        vae=load_run_vae(vae_run_dir, device=device),
        diffusion_model=load_denoiser(args, device=device),
        ddpm=build_ddpm(args, device=device),
        critic=critic,
        device=device,
    )
