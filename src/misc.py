import argparse
import math
from numbers import Real
from pathlib import Path

import torch
import yaml


def load_config(
    config_path: str | Path,
    *,
    label: str = "config file",
) -> dict:
    return _flatten_config(load_mapping(config_path, label=label))


def load_mapping(
    config_path: str | Path,
    *,
    label: str = "config file",
) -> dict:
    return _load_mapping(config_path, label=label)


def save_config(run_dir: str | Path, args: argparse.Namespace, name: str) -> None:
    if not name:
        raise ValueError("config name is required.")
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    config = {key: _encode_yaml(value) for key, value in vars(args).items()}
    with open(path / f"{name}.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def _load_mapping(config_path: str | Path, *, label: str) -> dict:
    try:
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is malformed: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{label} must contain a mapping.")
    return config


def _flatten_config(config: dict) -> dict:
    defaults = {}

    def visit(values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, dict) and key != "data_dir":
                visit(value)
            elif key in defaults:
                raise ValueError(f"Duplicate config key: {key}")
            else:
                defaults[key] = value

    visit(config)
    return defaults


def _encode_yaml(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_encode_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_yaml(item) for key, item in value.items()}
    return value


def require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def require_number(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def require_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
