import argparse
from pathlib import Path

import yaml

from src.app.api.options import (
    CriticConfig,
    JointConfig,
    PriorConfig,
    QualityConfig,
    RefineConfig,
    ScaleConfig,
    TargetConfig,
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


def load_predict_config(config_path: str | Path) -> dict:
    """Loads grouped prediction settings from YAML."""

    values = _load_mapping(config_path, label="prediction config")
    classes = {
        "prior": PriorConfig,
        "targets": TargetConfig,
        "joint": JointConfig,
        "critic": CriticConfig,
        "scale": ScaleConfig,
        "refine": RefineConfig,
        "quality": QualityConfig,
    }
    unknown = sorted(set(values) - set(classes) - {"progress"})
    if unknown:
        raise ValueError(f"prediction config contains unknown sections: {unknown}")

    config = {}
    if "progress" in values:
        if not isinstance(values["progress"], bool):
            raise ValueError("prediction config progress must be a boolean.")
        config["progress"] = values["progress"]
    for name, cls in classes.items():
        section = values.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"prediction config {name} must contain a mapping.")
        section = dict(section)
        if name == "critic" and "betas" in section:
            section["betas"] = tuple(section["betas"])
        if name == "refine" and "candidates" in section:
            section["candidates"] = tuple(section["candidates"])
        try:
            config[name] = cls(**section)
        except TypeError as exc:
            raise ValueError(
                f"prediction config {name} contains an unknown setting: {exc}"
            ) from exc
    return config


def save_run_config(run_dir: str | Path, args: argparse.Namespace, name: str) -> None:
    if not name:
        raise ValueError("config name is required.")

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    config = {key: _to_yaml(value) for key, value in vars(args).items()}

    with open(path / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
