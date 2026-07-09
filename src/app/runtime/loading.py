import argparse
from pathlib import Path
from typing import TypeVar

import torch

from src.app.api import Predictor
from src.modeling.diffusion import TimeUNet
from src.pipelines.training.runtime import freeze_module
from src.modeling.vae import PatchVAE
from src.app.runtime.config import (
    _last_model_path,
    _require_config_value,
    _require_config_values,
    _require_file,
    fill_diffusion_defaults_from_run,
    load_config_defaults,
)
from src.app.runtime.factories import (
    _vae_model_from_args,
    build_diffusion_model,
    build_diffusion_process,
)

ModelT = TypeVar("ModelT", bound=torch.nn.Module)

def _load_model_checkpoint(
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


def _load_frozen_checkpoint(
    model: ModelT,
    checkpoint_path: str | Path,
    device: torch.device,
    *state_dict_keys: str,
    label: str = "model checkpoint",
) -> ModelT:
    checkpoint_path = _require_file(checkpoint_path, label)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _load_model_checkpoint(model, checkpoint, *state_dict_keys)
    except Exception as exc:
        raise ValueError(
            f"{label} could not be loaded for model: {checkpoint_path}"
        ) from exc

    freeze_module(model)
    return model


def load_frozen_vae(args: argparse.Namespace, device: torch.device) -> PatchVAE:
    if getattr(args, "vae_ckpt", None) is None:
        raise ValueError("vae_ckpt is required.")

    vae = _vae_model_from_args(args).to(device)
    return _load_frozen_checkpoint(
        vae,
        args.vae_ckpt,
        device,
        "model",
        "vae",
        label="vae checkpoint",
    )


def load_frozen_vae_from_run(
    run_dir: str | Path,
    device: torch.device,
) -> PatchVAE:
    run_dir = Path(run_dir)
    vae_config = load_config_defaults(
        _require_file(run_dir / "vae.yaml", "vae config"),
        label="vae config",
    )
    _require_config_value(vae_config, "vae config", "image_size", "size")
    _require_config_value(vae_config, "vae config", "latent_ch")
    _require_config_value(vae_config, "vae config", "num_phases")
    args = argparse.Namespace(**vae_config)
    args.vae_ckpt = _last_model_path(run_dir, "vae")
    return load_frozen_vae(args, device=device)


def load_frozen_diffusion_model(
    args: argparse.Namespace,
    device: torch.device,
) -> TimeUNet:
    if getattr(args, "diffusion_ckpt", None) is None:
        raise ValueError("diffusion_ckpt is required.")

    model = build_diffusion_model(args).to(device)
    return _load_frozen_checkpoint(
        model,
        args.diffusion_ckpt,
        device,
        "model",
        "diffusion",
        "unet",
        label="diffusion checkpoint",
    )


def build_predictor_from_run(
    run_dir: str | Path,
    device: torch.device,
) -> Predictor:
    run_dir = Path(run_dir)
    diffusion_config = load_config_defaults(
        _require_file(run_dir / "diffusion.yaml", "diffusion config"),
        label="diffusion config",
    )
    _require_config_values(
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
    fill_diffusion_defaults_from_run(args)
    args.diffusion_ckpt = _last_model_path(run_dir, "diffusion")
    return Predictor(
        vae=load_frozen_vae_from_run(run_dir, device=device),
        diffusion_model=load_frozen_diffusion_model(args, device=device),
        ddpm=build_diffusion_process(args, device=device),
        device=device,
    )


def load_predictor(
    run_dir: str | Path,
    device: str | torch.device | None = None,
) -> Predictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_predictor_from_run(run_dir, device=torch.device(device))
