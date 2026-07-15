from collections.abc import Sequence
from dataclasses import dataclass

from ..model import MAX_PHASES
from ..misc import require_int, require_number


@dataclass(frozen=True)
class MPDDOptions:
    num_phases: int
    volume_size: int = 64
    phase_fractions: Sequence[float] | None = None
    segment_anchors: bool = False
    harmonization_steps: int = 10
    ddim_steps: int | None = None
    guidance_scale: float = 1.0
    tile_overlap: float = 0.25
    batch_size: int = 8
    progress: bool = True

    def __post_init__(self) -> None:
        require_int("num_phases", self.num_phases)
        require_int("volume_size", self.volume_size)
        require_int("harmonization_steps", self.harmonization_steps)
        if self.ddim_steps is not None:
            require_int("ddim_steps", self.ddim_steps)
        require_int("batch_size", self.batch_size)

        if self.num_phases < 2 or self.num_phases > MAX_PHASES:
            raise ValueError(f"num_phases must be between 2 and {MAX_PHASES}.")
        if self.volume_size <= 0:
            raise ValueError("volume_size must be positive.")
        if self.harmonization_steps <= 0:
            raise ValueError("harmonization_steps must be positive.")
        if self.ddim_steps is not None and self.ddim_steps <= 0:
            raise ValueError("ddim_steps must be positive or None.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        require_number("guidance_scale", self.guidance_scale)
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative.")
        if self.phase_fractions is None and self.guidance_scale != 1.0:
            raise ValueError(
                "phase_fractions are required when guidance_scale is not one."
            )
        require_number("tile_overlap", self.tile_overlap)
        if not 0.0 <= self.tile_overlap < 1.0:
            raise ValueError("tile_overlap must be at least zero and less than one.")

        if self.phase_fractions is not None:
            if isinstance(self.phase_fractions, (str, bytes)):
                raise ValueError("phase_fractions must be a sequence of numbers.")
            fractions = tuple(self.phase_fractions)
            if len(fractions) != self.num_phases:
                raise ValueError("phase_fractions must contain one value per phase.")
            for index, fraction in enumerate(fractions):
                require_number(f"phase_fractions[{index}]", fraction)
                if not 0.0 <= fraction <= 1.0:
                    raise ValueError(
                        "phase_fractions values must be between zero and one."
                    )
            if abs(sum(fractions) - 1.0) > 1e-4:
                raise ValueError("phase_fractions must sum to one.")
            object.__setattr__(self, "phase_fractions", tuple(map(float, fractions)))

        if not isinstance(self.segment_anchors, bool):
            raise ValueError("segment_anchors must be a boolean.")
        if not isinstance(self.progress, bool):
            raise ValueError("progress must be a boolean.")
