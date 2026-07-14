import argparse
from pathlib import Path
import shutil

from src.app.runtime.config import load_defaults


def checkpoint_path(run_dir: str | Path, component: str) -> Path:
    return Path(run_dir) / "weight" / component / "last" / "model.pt"


def require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is required: {path}")
    return path


def require_values(config: dict, label: str, *names: str) -> None:
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"{label} is missing required value: {', '.join(missing)}")


def load_run_config(run_dir: str | Path, name: str) -> dict:
    label = f"{name} config"
    path = require_file(Path(run_dir) / f"{name}.yaml", label)
    return load_defaults(path, label=label)


def copy_vae_run(source_run_dir: str | Path, target_run_dir: str | Path) -> None:
    source = Path(source_run_dir)
    target = Path(target_run_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "vae.yaml", target / "vae.yaml")

    source_weight = checkpoint_path(source, "vae")
    target_weight = checkpoint_path(target, "vae")
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weight, target_weight)


def apply_vae_defaults(args: argparse.Namespace) -> argparse.Namespace:
    vae_config = load_run_config(args.vae_run_dir, "vae")
    names = ("crop_size", "size", "segment", "num_phases", "latent_ch")
    require_values(vae_config, "vae config", *names)

    for arg_name in names:
        value = vae_config[arg_name]
        existing = getattr(args, arg_name, None)
        if existing is not None and existing != value:
            raise ValueError(f"{arg_name} must match VAE run config.")
        setattr(args, arg_name, value)

    latent_size = vae_config.get("latent_size")
    if latent_size is not None and int(latent_size) % 4 != 0:
        raise ValueError("latent_size must be divisible by 4 for diffusion.")
    return args


def apply_gan_defaults(args: argparse.Namespace) -> argparse.Namespace:
    vae_config = load_run_config(args.vae_run_dir, "vae")
    names = (
        "crop_size",
        "size",
        "segment",
        "num_phases",
        "latent_size",
        "latent_ch",
    )
    require_values(vae_config, "vae config", *names)
    for name in names:
        existing = getattr(args, name, None)
        value = vae_config[name]
        if existing is not None and existing != value:
            raise ValueError(f"{name} must match VAE run config.")
        setattr(args, name, value)
    return args
