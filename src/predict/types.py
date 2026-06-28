from dataclasses import dataclass

import numpy as np


MAX_UINT8_PHASES = int(np.iinfo(np.uint8).max) + 1


@dataclass(frozen=True)
class AnchorSlice:
    image: np.ndarray
    axis: int
    index: int


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
    sds_lr: float = 1e-2
    sds_t_min: int = 1
    sds_t_max: int | None = None
    sds_weight: float = 1.0

    refine_steps: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_phases, int)
            or isinstance(self.num_phases, bool)
        ):
            raise ValueError("num_phases must be an integer.")
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
            if weight < 0.0 or weight > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.diffusivity_low_cond < 0.0 or self.diffusivity_low_cond > 1.0:
            raise ValueError("diffusivity_low_cond must be between 0 and 1.")

        if self.sds_steps < 0:
            raise ValueError("sds_steps must be non-negative.")
        if self.sds_slice_steps < 0:
            raise ValueError("sds_slice_steps must be non-negative.")
        if self.sds_lr <= 0.0:
            raise ValueError("sds_lr must be positive.")
        if self.sds_t_min < 0:
            raise ValueError("sds_t_min must be non-negative.")
        if self.sds_t_max is not None and self.sds_t_max <= self.sds_t_min:
            raise ValueError("sds_t_max must be greater than sds_t_min.")
        if self.sds_weight < 0.0 or self.sds_weight > 1.0:
            raise ValueError("sds_weight must be between 0 and 1.")

        if self.refine_steps < 0:
            raise ValueError("refine_steps must be non-negative.")
