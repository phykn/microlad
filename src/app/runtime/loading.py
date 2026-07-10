import argparse
from pathlib import Path
from typing import TypeVar

import torch

from src.app.api import Predictor
from src.modeling.diffusion import TimeUNet
from src.pipelines.training.runtime import freeze_module
from src.modeling.vae import PatchVAE
from src.app.runtime.config import (
    _last_checkpoint,
    _require_value,
    _require_values,
    _require_file,
    apply_vae_defaults,
    load_defaults,
)
from src.app.runtime.factories import (
    _make_vae,
    build_denoiser,
    build_ddpm,
)

ModelT = TypeVar("ModelT", bound=torch.nn.Module)

def _load_state(
    target: torch.nn.Module,
    checkpoint,
    *state_dict_keys: str,
) -> None:
    if isinstance(checkpoint, dict):
        for key in state_dict_keys:
            if key in checkpoint:
                target.load_state_dict(checkpoint[key])
                return

    target.load_state_dict(checkpoint)


def _load_frozen(
    model: ModelT,
    checkpoint_path: str | Path,
    device: torch.device,
    *state_dict_keys: str,
    label: str = "model checkpoint",
) -> ModelT:
    checkpoint_path = _require_file(checkpoint_path, label)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _load_state(model, checkpoint, *state_dict_keys)
    except Exception as exc:
        raise ValueError(
            f"{label} could not be loaded for model: {checkpoint_path}"
        ) from exc

    freeze_module(model)
    return model


def load_frozen_vae(args: argparse.Namespace, device: torch.device) -> PatchVAE:
    if getattr(args, "vae_ckpt", None) is None:
        raise ValueError("vae_ckpt is required.")

    vae = _make_vae(args).to(device)
    return _load_frozen(
        vae,
        args.vae_ckpt,
        device,
        "model",
        "vae",
        label="vae checkpoint",
    )


def load_run_vae(
    run_dir: str | Path,
    device: torch.device,
) -> PatchVAE:
    run_dir = Path(run_dir)
    vae_config = load_defaults(
        _require_file(run_dir / "vae.yaml", "vae config"),
        label="vae config",
    )
    _require_value(vae_config, "vae config", "image_size", "size")
    _require_value(vae_config, "vae config", "latent_ch")
    _require_value(vae_config, "vae config", "num_phases")
    args = argparse.Namespace(**vae_config)
    args.vae_ckpt = _last_checkpoint(run_dir, "vae")
    return load_frozen_vae(args, device=device)


def load_denoiser(
    args: argparse.Namespace,
    device: torch.device,
) -> TimeUNet:
    if getattr(args, "diffusion_ckpt", None) is None:
        raise ValueError("diffusion_ckpt is required.")

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


def build_predictor(
    run_dir: str | Path,
    device: torch.device,
) -> Predictor:
    run_dir = Path(run_dir)
    diffusion_config = load_defaults(
        _require_file(run_dir / "diffusion.yaml", "diffusion config"),
        label="diffusion config",
    )
    _require_values(
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
    args.diffusion_ckpt = _last_checkpoint(run_dir, "diffusion")
    return Predictor(
        vae=load_run_vae(run_dir, device=device),
        diffusion_model=load_denoiser(args, device=device),
        ddpm=build_ddpm(args, device=device),
        device=device,
    )


def load_predictor(
    run_dir: str | Path,
    device: str | torch.device | None = None,
) -> Predictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_predictor(run_dir, device=torch.device(device))
