import argparse
from pathlib import Path

import yaml

from src.app.api.options import (
    CriticConfig,
    JointConfig,
    PriorConfig,
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


def load_predict_config(
    config_path: str | Path,
) -> tuple[dict[str, str | None], dict]:
    """Loads model run paths and grouped prediction settings from YAML."""

    values = _load_mapping(config_path, label="prediction config")
    models = values.pop("models", None)
    if not isinstance(models, dict):
        raise ValueError("prediction config models must contain a mapping.")
    model_names = ("vae_run_dir", "diffusion_run_dir", "gan_run_dir")
    unknown_models = sorted(set(models) - set(model_names))
    if unknown_models:
        raise ValueError(
            f"prediction config models contains unknown settings: {unknown_models}"
        )
    missing_models = sorted(set(model_names) - set(models))
    if missing_models:
        raise ValueError(
            f"prediction config models is missing settings: {missing_models}"
        )
    for name in ("vae_run_dir", "diffusion_run_dir"):
        if not isinstance(models[name], str) or not models[name].strip():
            raise ValueError(f"prediction config models.{name} is required.")
    if models["gan_run_dir"] is not None and (
        not isinstance(models["gan_run_dir"], str)
        or not models["gan_run_dir"].strip()
    ):
        raise ValueError(
            "prediction config models.gan_run_dir must be a path or null."
        )

    classes = {
        "prior": PriorConfig,
        "targets": TargetConfig,
        "joint": JointConfig,
        "critic": CriticConfig,
        "scale": ScaleConfig,
        "refine": RefineConfig,
    }
    root_settings = {
        "phase_fractions",
        "progress",
        "segment_anchors",
    }
    unknown = sorted(set(values) - set(classes) - root_settings)
    if unknown:
        raise ValueError(f"prediction config contains unknown sections: {unknown}")

    config = {
        name: values[name]
        for name in root_settings
        if name in values
    }
    for name in ("progress", "segment_anchors"):
        if name in config and not isinstance(config[name], bool):
            raise ValueError(f"prediction config {name} must be a boolean.")
    if "phase_fractions" in config and config["phase_fractions"] is not None:
        fractions = config["phase_fractions"]
        if not isinstance(fractions, (list, tuple)):
            raise ValueError("prediction config phase_fractions must be a sequence.")
        config["phase_fractions"] = tuple(fractions)
    for name, cls in classes.items():
        section = values.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"prediction config {name} must contain a mapping.")
        section = dict(section)
        try:
            config[name] = cls(**section)
        except TypeError as exc:
            raise ValueError(
                f"prediction config {name} contains an unknown setting: {exc}"
            ) from exc
    return {name: models[name] for name in model_names}, config


def save_run_config(run_dir: str | Path, args: argparse.Namespace, name: str) -> None:
    if not name:
        raise ValueError("config name is required.")

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    config = {key: _to_yaml(value) for key, value in vars(args).items()}

    with open(path / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
