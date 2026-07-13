from collections.abc import Sequence
from dataclasses import dataclass, field

from src.common.validation import require_finite_number, require_int
from src.modeling.phases.quantization import MAX_UINT8_PHASES
from src.pipelines.guidance.config import SliceGANConfig


def _require_weight(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_non_negative(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class AnchorConfig:
    segment: bool = False
    weight: float = 1.0
    fit_steps: int = 0
    fit_lr: float = 1e-1
    slab_radius: int = 2
    slab_weight: float = 0.25
    latent_sigma: float = 0.0
    latent_strength: float = 1.0
    axis_consensus: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.segment, bool):
            raise ValueError("segment must be a boolean.")
        _require_weight("weight", self.weight)
        require_int("fit_steps", self.fit_steps)
        if self.fit_steps < 0:
            raise ValueError("fit_steps must be non-negative.")
        require_finite_number("fit_lr", self.fit_lr)
        if self.fit_lr <= 0.0:
            raise ValueError("fit_lr must be positive.")
        require_int("slab_radius", self.slab_radius)
        if self.slab_radius < 0:
            raise ValueError("slab_radius must be non-negative.")
        _require_weight("slab_weight", self.slab_weight)
        _require_non_negative("latent_sigma", self.latent_sigma)
        require_finite_number("latent_strength", self.latent_strength)
        if self.latent_strength <= 0.0 or self.latent_strength > 1.0:
            raise ValueError("latent_strength must be greater than 0 and at most 1.")
        if not isinstance(self.axis_consensus, bool):
            raise ValueError("axis_consensus must be a boolean.")


@dataclass(frozen=True)
class TargetConfig:
    segment: bool = False
    vf_weight: float = 0.0
    tpc_weight: float = 0.0
    surface_weight: float = 0.0
    diffusivity_weight: float = 0.0
    diffusivity_size: int | tuple[int, int] | None = None
    low_conductivity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.segment, bool):
            raise ValueError("segment must be a boolean.")
        for name in (
            "vf_weight",
            "tpc_weight",
            "surface_weight",
            "diffusivity_weight",
        ):
            _require_weight(name, getattr(self, name))
        _require_weight("low_conductivity", self.low_conductivity)


@dataclass(frozen=True)
class SDSConfig:
    steps: int = 0
    slice_steps: int = 1
    batch_size: int = 1
    balanced_slices: bool = False
    consensus: bool = True
    learning_rate: float = 1e-2
    t_min: int = 1
    t_max: int | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        require_int("steps", self.steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")
        require_int("slice_steps", self.slice_steps)
        if self.slice_steps < 0:
            raise ValueError("slice_steps must be non-negative.")
        require_int("batch_size", self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(self.balanced_slices, bool):
            raise ValueError("balanced_slices must be a boolean.")
        if not isinstance(self.consensus, bool):
            raise ValueError("consensus must be a boolean.")
        require_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        require_int("t_min", self.t_min)
        if self.t_min < 0:
            raise ValueError("t_min must be non-negative.")
        if self.t_max is not None:
            require_int("t_max", self.t_max)
            if self.t_max <= self.t_min:
                raise ValueError("t_max must be greater than t_min.")
        _require_weight("weight", self.weight)


@dataclass(frozen=True)
class JointConfig:
    steps: int = 0
    batch_size: int = 8
    learning_rate: float = 1e-4
    entropy_weight: float = 1e-2
    continuity_weight: float = 5e-2
    transition_weight: float = 0.0
    run_weight: float = 0.0
    patch_weight: float = 0.0
    texture_weight: float = 0.0
    interface_weight: float = 0.0
    discriminator_lr: float = 1e-4

    def __post_init__(self) -> None:
        require_int("steps", self.steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")
        require_int("batch_size", self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        require_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        for name in (
            "entropy_weight",
            "continuity_weight",
            "transition_weight",
            "run_weight",
            "patch_weight",
            "texture_weight",
            "interface_weight",
        ):
            _require_non_negative(name, getattr(self, name))
        require_finite_number("discriminator_lr", self.discriminator_lr)
        if self.discriminator_lr <= 0.0:
            raise ValueError("discriminator_lr must be positive.")


@dataclass(frozen=True)
class RefineConfig:
    steps: int = 0

    def __post_init__(self) -> None:
        require_int("steps", self.steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")


@dataclass(frozen=True)
class PredictOptions:
    num_phases: int
    phase_fractions: Sequence[float] | None = None
    phase_fraction_tolerance: float = 0.01
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    sds: SDSConfig = field(default_factory=SDSConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    slicegan: SliceGANConfig | None = None
    refine: RefineConfig = field(default_factory=RefineConfig)

    def __post_init__(self) -> None:
        require_int("num_phases", self.num_phases)
        if self.num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        if self.num_phases > MAX_UINT8_PHASES:
            raise ValueError(
                f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 output."
            )
        if self.phase_fractions is not None:
            if isinstance(self.phase_fractions, (str, bytes)):
                raise ValueError("phase_fractions must be a sequence.")
            fractions = tuple(self.phase_fractions)
            if len(fractions) != self.num_phases:
                raise ValueError("phase_fractions must contain one value per phase.")
            for index, fraction in enumerate(fractions):
                require_finite_number(f"phase_fractions[{index}]", fraction)
                if fraction < 0.0 or fraction > 1.0:
                    raise ValueError("phase_fractions values must be between 0 and 1.")
            if abs(sum(fractions) - 1.0) > 1e-4:
                raise ValueError("phase_fractions must sum to one.")
            object.__setattr__(self, "phase_fractions", tuple(map(float, fractions)))
        require_finite_number(
            "phase_fraction_tolerance",
            self.phase_fraction_tolerance,
        )
        if self.phase_fraction_tolerance < 0.0 or self.phase_fraction_tolerance > 1.0:
            raise ValueError("phase_fraction_tolerance must be between 0 and 1.")
        for name, value, expected in (
            ("anchor", self.anchor, AnchorConfig),
            ("targets", self.targets, TargetConfig),
            ("sds", self.sds, SDSConfig),
            ("joint", self.joint, JointConfig),
            ("refine", self.refine, RefineConfig),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}.")
        if self.slicegan is not None and not isinstance(self.slicegan, SliceGANConfig):
            raise TypeError("slicegan must be SliceGANConfig or None.")
        if self.slicegan is not None and (self.sds.steps > 0 or self.joint.steps > 0):
            raise ValueError("slicegan cannot be combined with sds or joint.")
        if self.sds.steps > 0 and self.joint.steps > 0:
            raise ValueError("sds and joint cannot both be enabled.")
        if self.joint.steps > 0 and self.refine.steps > 0:
            raise ValueError("joint replaces refine.")
        if self.joint.steps > 0 and self.anchor.fit_steps > 0:
            raise ValueError("joint replaces anchor fitting.")
        if self.slicegan is not None and (
            self.refine.steps > 0 or self.anchor.fit_steps > 0
        ):
            raise ValueError("slicegan replaces refine and anchor fitting.")
