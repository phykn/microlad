from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..misc import load_mapping, require_int
from .options import MPDDOptions


class _Predictor(Protocol):
    image_size: int
    num_phases: int


@dataclass(frozen=True)
class PredictConfig:
    run_dir: Path
    generation: dict
    scale: dict

    def make_options(self, predictor: _Predictor) -> MPDDOptions:
        image_size = predictor.image_size
        num_phases = predictor.num_phases
        require_int("image_size", image_size)
        require_int("num_phases", num_phases)
        enabled = self.scale.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("scale.enabled must be a boolean.")

        values = dict(self.generation)
        values["num_phases"] = num_phases
        values["volume_size"] = (
            _require(self.scale, "volume_size") if enabled else image_size
        )
        values["tile_overlap"] = (
            _require(self.scale, "tile_overlap") if enabled else 0.0
        )
        return MPDDOptions(**values)


def load_predict_config(path: str | Path) -> PredictConfig:
    values = load_mapping(path, label="prediction config")
    model = _section(values, "model")
    run_dir = _require(model, "run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("model.run_dir must be a non-empty path.")
    return PredictConfig(
        run_dir=Path(run_dir),
        generation=_section(values, "generation"),
        scale=_section(values, "scale"),
    )


def _section(values: dict, name: str) -> dict:
    section = values.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"prediction config requires a {name} mapping.")
    return dict(section)


def _require(values: dict, name: str):
    if name not in values:
        raise ValueError(f"prediction config is missing {name}.")
    return values[name]
