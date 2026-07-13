import argparse
from pathlib import Path

import yaml

from src.app.api.options import (
    SliceGANConditionConfig,
    SliceGANConfig,
    SliceGANRenderConfig,
    SliceGANTrainConfig,
)


def _load_mapping(config_path: str | Path, *, label: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is malformed: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{label} must contain a mapping.")
    return config


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


def _to_yaml(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple)):
        return [_to_yaml(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _to_yaml(item) for key, item in value.items()}

    return value


def load_defaults(
    config_path: str | Path,
    *,
    label: str = "config file",
) -> dict:
    return _flatten_config(_load_mapping(config_path, label=label))


def load_slicegan_config(config_path: str | Path) -> SliceGANConfig:
    """Load the grouped conditional 3D generation settings from YAML."""

    values = _load_mapping(config_path, label="SliceGAN config")
    train = values.pop("train", {})
    condition = values.pop("condition", {})
    render = values.pop("render", {})
    for name, section in (
        ("train", train),
        ("condition", condition),
        ("render", render),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"SliceGAN config {name} must contain a mapping.")
    try:
        return SliceGANConfig(
            train=SliceGANTrainConfig(**train),
            condition=SliceGANConditionConfig(**condition),
            render=SliceGANRenderConfig(**render),
            **values,
        )
    except TypeError as exc:
        raise ValueError(f"SliceGAN config contains an unknown setting: {exc}") from exc


def save_run_config(run_dir: str | Path, args: argparse.Namespace, name: str) -> None:
    if not name:
        raise ValueError("config name is required.")

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    config = {key: _to_yaml(value) for key, value in vars(args).items()}

    with open(path / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
