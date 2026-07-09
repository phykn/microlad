import argparse
import shutil
from pathlib import Path

import yaml

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
        (
            "crop_size",
            _require_config_value(vae_config, "vae config", "crop_size"),
        ),
        ("size", vae_size),
        (
            "segment",
            _require_config_value(vae_config, "vae config", "segment"),
        ),
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


