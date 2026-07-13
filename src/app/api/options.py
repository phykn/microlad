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
class DiffusionAnchorConfig:
    """Configures anchor conditioning for diffusion-based generation.

    Attributes:
        weight: Strength of the anchor constraint.
        latent_sigma: Standard deviation of anchor influence in latent space.
        latent_strength: Peak anchor strength in latent space.
        axis_consensus: Whether to merge initial samples from all three axes.
        slab_radius: Number of neighboring slices influenced by an anchor.
        slab_weight: Blend ratio applied inside the neighboring slab.
        fit_steps: Number of post-generation anchor fitting steps.
        fit_lr: Learning rate used during anchor fitting.
    """

    weight: float = 1.0

    latent_sigma: float = 0.0
    latent_strength: float = 1.0
    axis_consensus: bool = False

    slab_radius: int = 0
    slab_weight: float = 0.0

    fit_steps: int = 0
    fit_lr: float = 1e-1

    def __post_init__(self) -> None:
        _require_non_negative("weight", self.weight)
        _require_non_negative("latent_sigma", self.latent_sigma)
        require_finite_number("latent_strength", self.latent_strength)
        if self.latent_strength <= 0.0 or self.latent_strength > 1.0:
            raise ValueError("latent_strength must be greater than 0 and at most 1.")
        if not isinstance(self.axis_consensus, bool):
            raise ValueError("axis_consensus must be a boolean.")
        require_int("slab_radius", self.slab_radius)
        if self.slab_radius < 0:
            raise ValueError("slab_radius must be non-negative.")
        _require_unit_interval("slab_weight", self.slab_weight)
        require_int("fit_steps", self.fit_steps)
        if self.fit_steps < 0:
            raise ValueError("fit_steps must be non-negative.")
        require_finite_number("fit_lr", self.fit_lr)
        if self.fit_lr <= 0.0:
            raise ValueError("fit_lr must be positive.")


@dataclass(frozen=True)
class PriorConfig:
    """Configures the shared diffusion prior used by SDS and Joint.

    Attributes:
        weight: Strength of the diffusion prior loss.
        t_min: Lowest diffusion timestep sampled during optimization.
        t_max: Exclusive upper timestep, or the full schedule when omitted.
    """

    weight: float = 1.0
    t_min: int = 1
    t_max: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative("weight", self.weight)
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
        vf_weight: Weight of the volume-fraction loss. When phase fractions are
            provided, zero selects the default weight of 1.0.
        tpc_weight: Weight of the two-point-correlation loss.
        surface_area_weight: Weight of the surface-area loss.
        diffusivity_weight: Weight of the effective-diffusivity loss.
        diffusivity_grid_size: Grid size used by the diffusivity solver.
        low_phase_conductivity: Conductivity assigned to the low phase.
    """

    segment: bool = False

    vf_weight: float = 0.0
    tpc_weight: float = 0.0
    surface_area_weight: float = 0.0
    diffusivity_weight: float = 0.0

    diffusivity_grid_size: int | tuple[int, int] | None = None
    low_phase_conductivity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.segment, bool):
            raise ValueError("segment must be a boolean.")
        for name in (
            "vf_weight",
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
class SDSConfig:
    """Configures slice-wise score distillation.

    Attributes:
        steps: Number of volume update steps. Zero disables SDS.
        slice_steps: Optimization steps applied to each selected slice batch.
        batch_size: Number of slices optimized together.
        balanced_slices: Whether to visit every axis and index evenly.
        consensus_sweeps: Whether to merge categorical predictions per sweep.
        learning_rate: Learning rate for slice optimization.
    """

    steps: int = 0
    slice_steps: int = 1
    batch_size: int = 1

    balanced_slices: bool = False
    consensus_sweeps: bool = True

    learning_rate: float = 1e-2

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
        if not isinstance(self.consensus_sweeps, bool):
            raise ValueError("consensus_sweeps must be a boolean.")
        require_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True)
class JointConfig:
    """Configures joint optimization of the complete 3D volume.

    Attributes:
        steps: Number of joint update steps. Zero disables Joint.
        batch_size: Number of slices sampled per update.
        learning_rate: Learning rate for the 3D generator.
        entropy_weight: Weight encouraging confident phase assignments.
        continuity_weight: Weight encouraging continuity between slices.
        transition_weight: Weight matching transition statistics.
        run_weight: Weight matching phase run-length statistics.
        patch_weight: Weight of the reference patch discriminator.
        texture_weight: Weight matching reference texture features.
        interface_weight: Weight matching phase interfaces.
        discriminator_lr: Learning rate for the patch discriminator.
    """

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
class SliceGANTrainConfig:
    """Configures SliceGAN texture training.

    Attributes:
        steps: Number of primary GAN training steps.
        mix_steps: Additional steps mixing diffusion references.
        critic_steps: Critic updates per generator update.
        batch_size: Number of 2D slices per critic update.
        lr: Learning rate shared by generator and critic.
        betas: Adam momentum coefficients.
        gradient_weight: Weight of the WGAN gradient penalty.
        reference_count: Number of diffusion reference images.
        mix_probability: Probability of sampling diffusion references.
        preview_count: Number of fixed noises used to score checkpoints.
    """

    steps: int = 5000
    mix_steps: int = 1000
    critic_steps: int = 5
    batch_size: int = 8

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    gradient_weight: float = 10.0

    reference_count: int = 8
    mix_probability: float = 0.1
    preview_count: int = 3

    def __post_init__(self) -> None:
        _require_non_negative_int("steps", self.steps)
        if self.steps == 0:
            raise ValueError("steps must be positive.")
        _require_non_negative_int("mix_steps", self.mix_steps)
        _require_non_negative_int("critic_steps", self.critic_steps)
        if self.critic_steps == 0:
            raise ValueError("critic_steps must be positive.")
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")

        _require_positive("lr", self.lr)
        if len(self.betas) != 2:
            raise ValueError("betas must contain two values.")
        for index, beta in enumerate(self.betas):
            require_finite_number(f"betas[{index}]", beta)
            if beta < 0.0 or beta >= 1.0:
                raise ValueError("betas must be between 0 inclusive and 1 exclusive.")
        object.__setattr__(self, "betas", tuple(map(float, self.betas)))
        _require_non_negative("gradient_weight", self.gradient_weight)

        _require_non_negative_int("reference_count", self.reference_count)
        if self.reference_count == 0:
            raise ValueError("reference_count must be positive.")
        _require_unit_interval("mix_probability", self.mix_probability)
        _require_non_negative_int("preview_count", self.preview_count)
        if self.preview_count == 0:
            raise ValueError("preview_count must be positive.")


@dataclass(frozen=True)
class SliceGANConditionConfig:
    """Configures conditional SliceGAN anchor fitting.

    Attributes:
        noise_steps: Number of noise optimization steps.
        tune_steps: Number of generator fine-tuning steps.
        candidates: Candidate noise volumes evaluated per condition.
        min_trials: Trained generators evaluated before quality-based stopping.
        noise_lr: Learning rate for initial noise optimization.
        critic_weight: Critic-prior weight during noise optimization.
        generator_lr: Generator learning rate during fine-tuning.
        tune_noise_lr: Noise learning rate during fine-tuning.
        tune_critic_weight: Critic-prior weight during fine-tuning.
        phase_weight: Phase-fraction weight during fine-tuning.
        preserve_weight: Weight preserving the initial volume.
        transition_weight: Weight matching anchor transitions.
        influence_sigma: Spatial spread of anchor influence.
        mismatch_tolerance: Desired mismatch ratio at anchor slices.
        morphology_tolerance: Allowed transition and run-profile error.
        continuity_tolerance: Allowed local boundary variation.
    """

    noise_steps: int = 800
    tune_steps: int = 500
    candidates: int = 8
    min_trials: int = 3

    noise_lr: float = 5e-2
    critic_weight: float = 1e-2

    generator_lr: float = 1e-5
    tune_noise_lr: float = 2e-3
    tune_critic_weight: float = 2e-2
    phase_weight: float = 50.0
    preserve_weight: float = 5.0

    transition_weight: float = 5.0
    influence_sigma: float = 8.0
    mismatch_tolerance: float = 0.08
    morphology_tolerance: float = 0.05
    continuity_tolerance: float = 0.08

    def __post_init__(self) -> None:
        _require_non_negative_int("noise_steps", self.noise_steps)
        _require_non_negative_int("tune_steps", self.tune_steps)
        _require_non_negative_int("candidates", self.candidates)
        if self.candidates == 0:
            raise ValueError("candidates must be positive.")
        _require_non_negative_int("min_trials", self.min_trials)
        if self.min_trials == 0:
            raise ValueError("min_trials must be positive.")

        for name in (
            "noise_lr",
            "generator_lr",
            "tune_noise_lr",
            "influence_sigma",
        ):
            _require_positive(name, getattr(self, name))
        for name in (
            "critic_weight",
            "tune_critic_weight",
            "phase_weight",
            "preserve_weight",
            "transition_weight",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_unit_interval("mismatch_tolerance", self.mismatch_tolerance)
        _require_unit_interval("morphology_tolerance", self.morphology_tolerance)
        _require_unit_interval("continuity_tolerance", self.continuity_tolerance)


@dataclass(frozen=True)
class SliceGANRenderConfig:
    """Configures tiled rendering of large SliceGAN volumes.

    Attributes:
        core_noise_size: Noise-tile core size.
        halo_noise_size: Overlap surrounding each noise tile.
    """

    core_noise_size: int = 4
    halo_noise_size: int = 4

    def __post_init__(self) -> None:
        _require_non_negative_int("core_noise_size", self.core_noise_size)
        if self.core_noise_size == 0:
            raise ValueError("core_noise_size must be positive.")
        _require_non_negative_int("halo_noise_size", self.halo_noise_size)


@dataclass(frozen=True)
class SliceGANConfig:
    """Collects settings for conditional SliceGAN generation.

    Attributes:
        train: Texture-training settings.
        condition: Anchor-conditioning settings.
        render: Large-volume rendering settings.
        intersection_tolerance: Allowed mismatch where anchors intersect.
    """

    train: SliceGANTrainConfig = field(default_factory=SliceGANTrainConfig)
    condition: SliceGANConditionConfig = field(default_factory=SliceGANConditionConfig)
    render: SliceGANRenderConfig = field(default_factory=SliceGANRenderConfig)

    intersection_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.train, SliceGANTrainConfig):
            raise TypeError("train must be SliceGANTrainConfig.")
        if not isinstance(self.condition, SliceGANConditionConfig):
            raise TypeError("condition must be SliceGANConditionConfig.")
        if not isinstance(self.render, SliceGANRenderConfig):
            raise TypeError("render must be SliceGANRenderConfig.")
        _require_unit_interval(
            "intersection_tolerance",
            self.intersection_tolerance,
        )


@dataclass(frozen=True)
class ScaleConfig:
    """Configures tiled scale-up operations.

    Attributes:
        overlap: Fraction of each tile shared with neighboring tiles.
        batch_size: Maximum number of scale-up planes or tiles processed together.
    """

    overlap: float = 0.25
    batch_size: int = 16

    def __post_init__(self) -> None:
        require_finite_number("overlap", self.overlap)
        if self.overlap < 0.0 or self.overlap >= 1.0:
            raise ValueError("overlap must be at least 0 and less than 1.")
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")


@dataclass(frozen=True)
class RefineConfig:
    """Configures VAE-based refinement after generation.

    Attributes:
        steps: Number of refinement passes. Zero disables refinement.
        batch_size: Maximum number of base-volume slices decoded together.
    """

    steps: int = 0
    batch_size: int = 16

    def __post_init__(self) -> None:
        _require_non_negative_int("steps", self.steps)
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")


@dataclass(frozen=True)
class PredictOptions:
    """Collects user-facing options for conditional 3D generation.

    Attributes:
        num_phases: Number of categorical material phases.
        phase_fractions: Desired fraction of each phase, ordered by label. SDS
            and Joint use a default volume-fraction weight of 1.0 when
            targets.vf_weight is zero.
        phase_fraction_tolerance: Allowed fraction error for SliceGAN output.
        segment_anchors: Whether to segment anchor images before conditioning.
        diffusion_anchor: Anchor settings for diffusion-based methods.
        prior: Diffusion prior shared by SDS and Joint.
        targets: Losses derived from reference images.
        sds: Slice-wise score-distillation settings.
        joint: Full-volume joint optimization settings.
        slicegan: Conditional SliceGAN settings, or None to use diffusion.
        scale: Tiled scale-up settings.
        refine: Optional VAE refinement settings.
    """

    num_phases: int
    phase_fractions: Sequence[float] | None = None
    phase_fraction_tolerance: float = 0.01

    segment_anchors: bool = False
    diffusion_anchor: DiffusionAnchorConfig = field(
        default_factory=DiffusionAnchorConfig
    )

    prior: PriorConfig = field(default_factory=PriorConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    sds: SDSConfig = field(default_factory=SDSConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    slicegan: SliceGANConfig | None = None
    scale: ScaleConfig = field(default_factory=ScaleConfig)
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
        if not isinstance(self.segment_anchors, bool):
            raise ValueError("segment_anchors must be a boolean.")
        for name, value, expected in (
            ("diffusion_anchor", self.diffusion_anchor, DiffusionAnchorConfig),
            ("prior", self.prior, PriorConfig),
            ("targets", self.targets, TargetConfig),
            ("sds", self.sds, SDSConfig),
            ("joint", self.joint, JointConfig),
            ("scale", self.scale, ScaleConfig),
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
        if self.joint.steps > 0 and self.diffusion_anchor.fit_steps > 0:
            raise ValueError("joint replaces anchor fitting.")
        if self.slicegan is not None and (
            self.refine.steps > 0 or self.diffusion_anchor.fit_steps > 0
        ):
            raise ValueError("slicegan replaces refine and anchor fitting.")
