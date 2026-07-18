from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TypeVar

from ..misc import load_mapping


@dataclass(frozen=True)
class DataConfig:
    data_dir: dict[int, str | Path]
    crop_size: int
    size: int
    num_phases: int
    batch_size: int
    segment: bool = False
    augment: bool = False
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    base_ch: int
    time_dim: int


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int
    beta_start: float
    beta_end: float


@dataclass(frozen=True)
class OptimizationConfig:
    lr: float
    weight_decay: float = 0.0
    clip_grad_norm: float | None = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    save_every: int
    ema_decay: float
    frac_dropout: float
    anchor_weight: float
    ckpt: str | Path | None = None
    warmup_steps: int = 0


@dataclass(frozen=True)
class OutputConfig:
    run_root: str | Path


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    optimization: OptimizationConfig
    training: TrainingConfig
    output: OutputConfig

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_T = TypeVar("_T")
_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "diffusion": DiffusionConfig,
    "optimization": OptimizationConfig,
    "training": TrainingConfig,
    "output": OutputConfig,
}
def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path).resolve()
    vals = load_mapping(path, label="training config")
    names = set(vals)
    expected = set(_SECTIONS)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        parts = []
        if missing:
            parts.append(f"missing sections: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(f"training config has {'; '.join(parts)}.")

    data = _make(DataConfig, vals["data"], "data")
    data = replace(data, data_dir=_resolve_dirs(data.data_dir, path.parent))
    training = _make(TrainingConfig, vals["training"], "training")
    training = replace(
        training,
        ckpt=_resolve_path(training.ckpt, path.parent, "training.ckpt"),
    )
    return TrainConfig(
        data=data,
        model=_make(ModelConfig, vals["model"], "model"),
        diffusion=_make(DiffusionConfig, vals["diffusion"], "diffusion"),
        optimization=_make(
            OptimizationConfig,
            vals["optimization"],
            "optimization",
        ),
        training=training,
        output=_make(OutputConfig, vals["output"], "output"),
    )


def _make(cls: type[_T], val: object, name: str) -> _T:
    if not isinstance(val, dict):
        raise ValueError(f"training config section {name} must be a mapping.")
    try:
        return cls(**val)
    except TypeError as exc:
        raise ValueError(f"training config section {name} is invalid: {exc}") from exc


def _resolve_dirs(val: object, root: Path) -> dict[int, Path]:
    if not isinstance(val, dict) or set(val) != {0, 1, 2}:
        raise ValueError("data.data_dir must contain exactly axes 0, 1, and 2.")
    return {
        axis: _resolve_dir(path, root, f"data.data_dir.{axis}")
        for axis, path in val.items()
    }


def _resolve_dir(val: object, root: Path, name: str) -> Path:
    path = _resolve_path(val, root, name)
    if path is None:
        raise ValueError(f"{name} must be a non-empty path.")
    return path


def _resolve_path(val: object, root: Path, name: str) -> Path | None:
    if val is None:
        return None
    if not isinstance(val, (str, Path)) or not str(val).strip():
        raise ValueError(f"{name} must be a non-empty path or null.")
    path = Path(val)
    return (path if path.is_absolute() else root / path).resolve()
