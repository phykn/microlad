import argparse
from pathlib import Path
from typing import TypeVar

import torch

from src.app.runtime.model import (
    build_critic,
    build_denoiser,
    build_generator,
    build_vae,
)
from src.app.runtime.run import (
    checkpoint_path,
    load_run_config,
    require_file,
    require_values,
)
from src.modeling.diffusion import TimeUNet
from src.modeling.latent_gan import ImageCritic, LatentGenerator
from src.modeling.vae import PatchVAE


ModelT = TypeVar("ModelT", bound=torch.nn.Module)


def _load_frozen(
    model: ModelT,
    checkpoint_path: str | Path,
    device: torch.device,
    *state_dict_keys: str,
    label: str,
) -> ModelT:
    checkpoint_path = require_file(checkpoint_path, label)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint
        if isinstance(checkpoint, dict):
            state = next(
                (checkpoint[key] for key in state_dict_keys if key in checkpoint),
                checkpoint,
            )
        model.load_state_dict(state)
    except Exception as exc:
        raise ValueError(
            f"{label} could not be loaded for model: {checkpoint_path}"
        ) from exc

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_run_vae(
    run_dir: str | Path,
    device: torch.device,
) -> PatchVAE:
    vae_config = load_run_config(run_dir, "vae")
    require_values(
        vae_config,
        "vae config",
        "size",
        "latent_size",
        "latent_ch",
        "num_phases",
        "base_ch",
        "max_ch",
    )
    args = argparse.Namespace(**vae_config)
    vae = build_vae(args).to(device)
    return _load_frozen(
        vae,
        checkpoint_path(run_dir, "vae"),
        device,
        "model",
        "vae",
        label="vae checkpoint",
    )


def load_denoiser(
    args: argparse.Namespace,
    device: torch.device,
) -> TimeUNet:
    model = build_denoiser(args).to(device)
    return _load_frozen(
        model,
        args.diffusion_ckpt,
        device,
        "model",
        "diffusion",
        "unet",
        label="diffusion checkpoint",
    )


def load_run_generator(
    run_dir: str | Path,
    device: torch.device,
) -> LatentGenerator:
    config = load_run_config(run_dir, "gan")
    require_values(
        config,
        "gan config",
        "latent_ch",
        "latent_size",
        "noise_ch",
        "generator_ch",
    )
    model = build_generator(argparse.Namespace(**config)).to(device)
    return _load_frozen(
        model,
        checkpoint_path(run_dir, "gan"),
        device,
        "generator",
        label="GAN generator checkpoint",
    )


def load_run_critic(
    run_dir: str | Path,
    device: torch.device,
) -> ImageCritic:
    config = load_run_config(run_dir, "gan")
    require_values(
        config,
        "gan config",
        "num_phases",
        "size",
        "critic_ch",
    )
    model = build_critic(argparse.Namespace(**config)).to(device)
    return _load_frozen(
        model,
        checkpoint_path(run_dir, "gan"),
        device,
        "critic",
        label="GAN critic checkpoint",
    )
