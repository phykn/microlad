from collections.abc import Sequence
from dataclasses import dataclass, field

from src.validation import require_finite_number, require_int
from src.modeling.phases.quantization import MAX_UINT8_PHASES


def _require_unit_interval(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_non_negative(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def _require_non_negative_int(name: str, value: int) -> None:
    require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_positive(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class PriorConfig:
    """Configures the diffusion prior used during latent refinement.

    Attributes:
        weight: Strength of the diffusion prior loss.
        anchor_strength: Soft latent anchor blend used during L-MPDD sampling.
        t_min: Lowest diffusion timestep sampled during optimization.
        t_max: Exclusive upper timestep, or the full schedule when omitted.
    """

    weight: float = 1.0
    anchor_strength: float = 0.0
    t_min: int = 1
    t_max: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative("weight", self.weight)
        _require_unit_interval("anchor_strength", self.anchor_strength)
        require_int("t_min", self.t_min)
        if self.t_min < 0:
            raise ValueError("t_min must be non-negative.")
        if self.t_max is not None:
            require_int("t_max", self.t_max)
            if self.t_max <= self.t_min:
                raise ValueError("t_max must be greater than t_min.")


@dataclass(frozen=True)
class TargetConfig:
    """Configures losses derived from reference images.

    Attributes:
        segment: Whether to segment reference images before analysis.
        slice_fraction_weight: Weight matching each sampled 2D slice fraction.
        global_fraction_weight: Weight matching the full 3D volume fraction.
        tpc_weight: Weight of the two-point-correlation loss.
        surface_area_weight: Weight of the surface-area loss.
        diffusivity_weight: Weight of the effective-diffusivity loss.
        diffusivity_grid_size: Grid size used by the diffusivity solver.
        low_phase_conductivity: Conductivity assigned to the low phase.
    """

    segment: bool = False

    slice_fraction_weight: float = 0.0
    global_fraction_weight: float = 0.0
    tpc_weight: float = 0.0
    surface_area_weight: float = 0.0
    diffusivity_weight: float = 0.0

    diffusivity_grid_size: int | tuple[int, int] | None = None
    low_phase_conductivity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.segment, bool):
            raise ValueError("segment must be a boolean.")
        for name in (
            "slice_fraction_weight",
            "global_fraction_weight",
            "tpc_weight",
            "surface_area_weight",
            "diffusivity_weight",
        ):
            _require_non_negative(name, getattr(self, name))

        grid_size = self.diffusivity_grid_size
        if self.diffusivity_weight > 0.0 and grid_size is None:
            raise ValueError(
                "diffusivity_grid_size is required when diffusivity_weight is positive."
            )
        if grid_size is not None:
            if isinstance(grid_size, int) and not isinstance(grid_size, bool):
                height = width = grid_size
            elif isinstance(grid_size, tuple) and len(grid_size) == 2:
                height, width = grid_size
            else:
                raise ValueError(
                    "diffusivity_grid_size must be an integer or (height, width)."
                )

            require_int("diffusivity_grid_size height", height)
            require_int("diffusivity_grid_size width", width)
            if height < 2 or width < 2:
                raise ValueError("diffusivity_grid_size dimensions must be at least 2.")

        _require_unit_interval(
            "low_phase_conductivity",
            self.low_phase_conductivity,
        )


@dataclass(frozen=True)
class JointConfig:
    """Configures latent joint optimization at the trained VAE size.

    Attributes:
        steps: Number of joint update steps. Zero disables Joint.
        batch_size: Number of slices sampled per update.
        decode_batch_size: Number of VAE planes decoded at once during Joint.
            Set to None to decode each axis in one batch without checkpointing.
        learning_rate: Learning rate for the 3D generator.
        axis_weight: Weight aligning decoded phase probabilities across axes.
        continuity_weight: Weight encouraging continuity between slices.
        anchor_weight: Weight of exact decoded anchor conditioning.
        residual_scale: Maximum residual in channel standard deviations.
        preservation_weight: Weight keeping the result close to L-MPDD.
        checkpoint_every: Interval between candidate checkpoints.
    """

    steps: int = 0
    batch_size: int = 8
    decode_batch_size: int | None = 16
    learning_rate: float = 1e-4

    axis_weight: float = 1.0
    continuity_weight: float = 5e-2
    anchor_weight: float = 1.0

    residual_scale: float = 0.25
    preservation_weight: float = 5.0
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        require_int("steps", self.steps)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")
        require_int("batch_size", self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.decode_batch_size is not None:
            require_int("decode_batch_size", self.decode_batch_size)
            if self.decode_batch_size <= 0:
                raise ValueError("decode_batch_size must be positive or None.")
        require_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        for name in (
            "axis_weight",
            "continuity_weight",
            "anchor_weight",
            "preservation_weight",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_positive("residual_scale", self.residual_scale)
        _require_non_negative_int("checkpoint_every", self.checkpoint_every)
        if self.checkpoint_every == 0:
            raise ValueError("checkpoint_every must be positive.")


@dataclass(frozen=True)
class CriticConfig:
    """Configures pretrained latent critic guidance.

    Attributes:
        weight: Critic guidance weight during latent refinement.
    """

    weight: float = 0.0

    def __post_init__(self) -> None:
        _require_non_negative("weight", self.weight)


@dataclass(frozen=True)
class ScaleConfig:
    """Configures latent scale-up guidance.

    Attributes:
        overlap: Fraction of each tile shared with neighboring tiles.
        steps: Number of scale-up guidance updates.
        batch_size: Number of latent crops guided per update.
        decode_batch_size: Maximum planes or tiles decoded together. Use None to
            process each decode stage in one batch on a large-memory GPU.
        learning_rate: Learning rate for the 3D latent residual.
        anchor_weight: Weight of decoded categorical anchor conditioning.
        continuity_weight: Weight preserving three-axis latent continuity.
        preservation_weight: Weight keeping the result near the initial L-MPDD.
        residual_scale: Maximum residual in latent channel standard deviations.
        checkpoint_every: Interval between scale-up candidate checkpoints.
    """

    overlap: float = 0.25
    steps: int = 0
    batch_size: int = 8
    decode_batch_size: int | None = 16
    learning_rate: float = 3e-3

    anchor_weight: float = 2.0
    continuity_weight: float = 0.05
    preservation_weight: float = 1.0
    residual_scale: float = 2.0
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        require_finite_number("overlap", self.overlap)
        if self.overlap < 0.0 or self.overlap >= 1.0:
            raise ValueError("overlap must be at least 0 and less than 1.")
        _require_non_negative_int("steps", self.steps)
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")
        if self.decode_batch_size is not None:
            _require_non_negative_int("decode_batch_size", self.decode_batch_size)
            if self.decode_batch_size == 0:
                raise ValueError("decode_batch_size must be positive or None.")
        _require_positive("learning_rate", self.learning_rate)
        for name in (
            "anchor_weight",
            "continuity_weight",
            "preservation_weight",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_positive("residual_scale", self.residual_scale)
        _require_non_negative_int("checkpoint_every", self.checkpoint_every)
        if self.checkpoint_every == 0:
            raise ValueError("checkpoint_every must be positive.")


@dataclass(frozen=True)
class RefineConfig:
    """Configures categorical VAE refinement candidates.

    Attributes:
        candidates: Refinement sweep counts evaluated after latent optimization.
        batch_size: Maximum number of base-volume slices decoded together.
        strength: Blend weight of each categorical projection.
        anchor_strength: Blend weight used inside anchor footprints.
    """

    candidates: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 16
    strength: float = 0.15
    anchor_strength: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple.")
        normalized = []
        for index, steps in enumerate(self.candidates):
            _require_non_negative_int(f"candidates[{index}]", steps)
            normalized.append(int(steps))
        object.__setattr__(self, "candidates", tuple(dict.fromkeys(normalized)))
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")
        _require_unit_interval("strength", self.strength)
        _require_unit_interval("anchor_strength", self.anchor_strength)


@dataclass(frozen=True)
class QualityConfig:
    """Configures final candidate feasibility checks.

    Attributes:
        anchor_tolerance: Maximum mismatch allowed for any anchor.
        morphology_tolerance: Maximum transition or run-profile error.
        continuity_tolerance: Maximum global boundary-profile jump.
        repeat_tolerance: Maximum exact adjacent-slice repetition rate.
        calibration_budget: Maximum voxel fraction changed by calibration.
    """

    anchor_tolerance: float = 0.08
    morphology_tolerance: float = 0.05
    continuity_tolerance: float = 0.08
    repeat_tolerance: float = 0.0
    calibration_budget: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "anchor_tolerance",
            "morphology_tolerance",
            "continuity_tolerance",
            "repeat_tolerance",
            "calibration_budget",
        ):
            _require_unit_interval(name, getattr(self, name))


@dataclass(frozen=True)
class PredictOptions:
    """Collects user-facing options for conditional 3D generation.

    Attributes:
        num_phases: Number of categorical material phases.
        phase_fractions: Desired global fraction of each phase, ordered by label.
        phase_fraction_tolerance: Allowed fraction error after final calibration.
        segment_anchors: Whether to segment anchor images before conditioning.
        progress: Whether to show prediction progress bars.
        prior: Diffusion prior used during latent refinement.
        targets: Losses derived from reference images.
        joint: Full-volume joint optimization settings.
        critic: Pretrained latent critic guidance used by Joint and scale-up.
        quality: Final candidate feasibility settings.
        scale: Tiled scale-up settings.
        refine: Optional VAE refinement settings.
    """

    num_phases: int
    phase_fractions: Sequence[float] | None = None
    phase_fraction_tolerance: float = 0.01

    segment_anchors: bool = False
    progress: bool = True
    prior: PriorConfig = field(default_factory=PriorConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

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
        if not isinstance(self.segment_anchors, bool):
            raise ValueError("segment_anchors must be a boolean.")
        if not isinstance(self.progress, bool):
            raise ValueError("progress must be a boolean.")
        for name, value, expected in (
            ("prior", self.prior, PriorConfig),
            ("targets", self.targets, TargetConfig),
            ("joint", self.joint, JointConfig),
            ("critic", self.critic, CriticConfig),
            ("scale", self.scale, ScaleConfig),
            ("refine", self.refine, RefineConfig),
            ("quality", self.quality, QualityConfig),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}.")
