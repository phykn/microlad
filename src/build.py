import argparse
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from src.data import PatchDataset
from src.loss import DiffusionLoss, VAELoss
from src.models import DDPM, PatchVAE, TimeUNet
from src.predict import Predictor
from src.train import DiffusionTrainer, VAETrainer
from src.train.utils import freeze_module


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MISSING = object()
ModelT = TypeVar("ModelT", bound=torch.nn.Module)


def _flatten_config(config: dict) -> dict:
    defaults = {}

    def visit(values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                visit(value)
                continue

            if key in defaults:
                raise ValueError(f"Duplicate config key: {key}")

            defaults[key] = value

    visit(config)
    return defaults


def _image_paths_from_dir(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def _last_model_path(run_dir: str | Path, component: str) -> Path:
    return Path(run_dir) / "weight" / component / "last" / "model.pt"


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{label} is required: {path}")

    return path


def _require_config_value(config: dict, label: str, *names: str):
    for name in names:
        if name in config:
            return config[name]
    raise ValueError(f"{label} is missing required value: {' or '.join(names)}")


def _require_config_values(config: dict, label: str, *names: str) -> None:
    missing = [name for name in names if name not in config]

    if missing:
        raise ValueError(
            f"{label} is missing required value: {', '.join(missing)}"
        )


def _get_arg(args: argparse.Namespace, *names: str, default=_MISSING):
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)

    if default is not _MISSING:
        return default

    raise AttributeError(f"missing config value: {' or '.join(names)}")


def _vae_model_from_args(args: argparse.Namespace) -> PatchVAE:
    return PatchVAE(
        image_size=_get_arg(args, "vae_image_size", "image_size", "size"),
        latent_size=_get_arg(args, "vae_latent_size", "latent_size", default=16),
        latent_ch=_get_arg(args, "vae_latent_ch", "latent_ch"),
        num_phases=_get_arg(args, "vae_num_phases", "num_phases", default=3),
        base_ch=_get_arg(args, "vae_base_ch", "base_ch", default=64),
        max_ch=_get_arg(args, "vae_max_ch", "max_ch", default=512),
    )


def _yaml_safe_value(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [_yaml_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return [_yaml_safe_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _yaml_safe_value(item) for key, item in value.items()}

    return value


def load_config_defaults(
    config_path: str | Path | None,
    *,
    label: str = "config file",
) -> dict:
    if not config_path:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is malformed: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{label} must contain a mapping.")

    return _flatten_config(config)


def save_run_config(run_dir: str | Path, args: argparse.Namespace, name: str) -> None:
    if not name:
        raise ValueError("config name is required.")

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    config = {key: _yaml_safe_value(value) for key, value in vars(args).items()}

    with open(path / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def copy_vae_run(source_run_dir: str | Path, target_run_dir: str | Path) -> None:
    source = Path(source_run_dir)
    target = Path(target_run_dir)

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "vae.yaml", target / "vae.yaml")

    source_weight = _last_model_path(source, "vae")
    target_weight = _last_model_path(target, "vae")
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weight, target_weight)


def fill_diffusion_defaults_from_run(args: argparse.Namespace) -> argparse.Namespace:
    run_dir = getattr(args, "vae_run_dir", None)

    if run_dir is None:
        return args

    vae_config = load_config_defaults(
        _require_file(Path(run_dir) / "vae.yaml", "vae config"),
        label="vae config",
    )
    vae_size = _require_config_value(vae_config, "vae config", "image_size", "size")

    for arg_name, value in (
        ("size", vae_size),
        (
            "num_phases",
            _require_config_value(vae_config, "vae config", "num_phases"),
        ),
        (
            "latent_ch",
            _require_config_value(vae_config, "vae config", "latent_ch"),
        ),
    ):
        existing = getattr(args, arg_name, None)

        if existing is not None and existing != value:
            raise ValueError(f"{arg_name} must match VAE run config.")

        setattr(args, arg_name, value)

    latent_size = vae_config.get("latent_size")

    if latent_size is not None and int(latent_size) % 4 != 0:
        raise ValueError("latent_size must be divisible by 4 for diffusion.")

    return args


def setup_device() -> tuple[torch.device, int, bool]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return device, local_rank, True


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_distributed(
    model: torch.nn.Module,
    local_rank: int,
    distributed: bool,
) -> torch.nn.Module:
    if not distributed:
        return model

    if next(model.parameters()).device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[local_rank])

    return DistributedDataParallel(model)


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


def build_dataset(args: argparse.Namespace) -> PatchDataset:
    image_paths = getattr(args, "image_paths", None)

    if image_paths is None:
        image_paths = _image_paths_from_dir(args.data_dir)
    elif isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    return PatchDataset(
        image_paths,
        crop_size=args.crop_size,
        size=args.size,
        num_phases=args.num_phases,
        segment=getattr(args, "segment", False),
        augment=getattr(args, "augment", False),
    )


def build_loader(
    dataset: PatchDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Iterator:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    while True:
        indices = torch.randint(0, len(dataset), (args.batch_size,)).tolist()
        batch = torch.stack([dataset[index] for index in indices])
        yield batch.pin_memory() if device.type == "cuda" else batch


def build_vae(args: argparse.Namespace) -> PatchVAE:
    return _vae_model_from_args(args)


def build_diffusion_model(args: argparse.Namespace) -> TimeUNet:
    return TimeUNet(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
    )


def build_ddpm(
    args: argparse.Namespace,
    device: torch.device,
) -> DDPM:
    return DDPM(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
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
        ddpm=build_ddpm(args, device=device),
        device=device,
    )


def load_predictor(
    run_dir: str | Path,
    device: str | torch.device | None = None,
) -> Predictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_predictor_from_run(run_dir, device=torch.device(device))


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )


def build_vae_trainer(
    model: torch.nn.Module,
    loader: Iterator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> VAETrainer:
    loss_fn = VAELoss(
        beta=args.beta,
        ssim_weight=args.ssim_weight,
        num_phases=args.num_phases,
    )
    return VAETrainer(
        model=model,
        dataloader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        run_root=getattr(args, "run_root", "run"),
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
    )


def build_diffusion_trainer(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    loader: Iterator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> DiffusionTrainer:
    return DiffusionTrainer(
        model=model,
        vae=vae,
        dataloader=loader,
        loss_fn=DiffusionLoss(build_ddpm(args, device=device)),
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        run_root=getattr(args, "run_root", "run"),
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
    )
