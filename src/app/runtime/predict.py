import argparse
from pathlib import Path

import torch

from src.app.api import Predictor
from src.app.runtime.load import load_denoiser, load_run_vae
from src.app.runtime.model import build_ddpm
from src.app.runtime.run import (
    apply_vae_defaults,
    checkpoint_path,
    load_run_config,
    require_values,
)


def load_predictor(
    run_dir: str | Path,
    device: str | torch.device | None = None,
) -> Predictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    diffusion_config = load_run_config(run_dir, "diffusion")
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
    args.vae_run_dir = run_dir
    apply_vae_defaults(args)
    args.diffusion_ckpt = checkpoint_path(run_dir, "diffusion")
    return Predictor(
        vae=load_run_vae(run_dir, device=device),
        diffusion_model=load_denoiser(args, device=device),
        ddpm=build_ddpm(args, device=device),
        device=device,
    )
