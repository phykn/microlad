from dataclasses import dataclass
import math
from numbers import Real

from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.modeling.phases.quantization import MAX_UINT8_PHASES


@dataclass(frozen=True)
class PredictOptions:
    num_phases: int

    anchor_segment: bool = False
    anchor_weight: float = 1.0

    target_segment: bool = False
    vf_weight: float = 0.0
    tpc_weight: float = 0.0
    sa_weight: float = 0.0
    diffusivity_weight: float = 0.0

    diffusivity_size: int | tuple[int, int] | None = None
    diffusivity_low_cond: float = 0.0

    sds_steps: int = 0
    sds_slice_steps: int = 1
    sds_batch_size: int = 1
    sds_lr: float = 1e-2
    sds_t_min: int = 1
    sds_t_max: int | None = None
    sds_weight: float = 1.0

    refine_steps: int = 0

    def __post_init__(self) -> None:
        _validate_integer("num_phases", self.num_phases)

        if self.num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        if self.num_phases > MAX_UINT8_PHASES:
            raise ValueError(
                f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 output."
            )

        for name, weight in (
            ("anchor_weight", self.anchor_weight),
            ("vf_weight", self.vf_weight),
            ("tpc_weight", self.tpc_weight),
            ("sa_weight", self.sa_weight),
            ("diffusivity_weight", self.diffusivity_weight),
        ):
            _validate_finite_scalar(name, weight)

            if weight < 0.0 or weight > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        _validate_finite_scalar("diffusivity_low_cond", self.diffusivity_low_cond)

        if self.diffusivity_low_cond < 0.0 or self.diffusivity_low_cond > 1.0:
            raise ValueError("diffusivity_low_cond must be between 0 and 1.")

        _validate_integer("sds_steps", self.sds_steps)
        if self.sds_steps < 0:
            raise ValueError("sds_steps must be non-negative.")

        _validate_integer("sds_slice_steps", self.sds_slice_steps)
        if self.sds_slice_steps < 0:
            raise ValueError("sds_slice_steps must be non-negative.")

        _validate_integer("sds_batch_size", self.sds_batch_size)
        if self.sds_batch_size <= 0:
            raise ValueError("sds_batch_size must be positive.")

        _validate_finite_scalar("sds_lr", self.sds_lr)
        if self.sds_lr <= 0.0:
            raise ValueError("sds_lr must be positive.")

        _validate_integer("sds_t_min", self.sds_t_min)
        if self.sds_t_min < 0:
            raise ValueError("sds_t_min must be non-negative.")

        if self.sds_t_max is not None:
            _validate_integer("sds_t_max", self.sds_t_max)

            if self.sds_t_max <= self.sds_t_min:
                raise ValueError("sds_t_max must be greater than sds_t_min.")

        _validate_finite_scalar("sds_weight", self.sds_weight)
        if self.sds_weight < 0.0 or self.sds_weight > 1.0:
            raise ValueError("sds_weight must be between 0 and 1.")

        _validate_integer("refine_steps", self.refine_steps)
        if self.refine_steps < 0:
            raise ValueError("refine_steps must be non-negative.")


def _validate_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def _validate_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
