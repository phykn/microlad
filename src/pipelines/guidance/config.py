from dataclasses import dataclass, field

from src.common.validation import require_finite_number, require_int


def _require_non_negative_int(name: str, value: int) -> None:
    require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_positive_number(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _require_non_negative_number(name: str, value: float) -> None:
    require_finite_number(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class SliceGANTrainConfig:
    steps: int = 5000
    hybrid_steps: int = 1000
    critic_iterations: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    gradient_penalty_weight: float = 10.0
    diffusion_reference_count: int = 8
    diffusion_mix_probability: float = 0.1

    def __post_init__(self) -> None:
        _require_non_negative_int("steps", self.steps)
        if self.steps == 0:
            raise ValueError("steps must be positive.")
        _require_non_negative_int("hybrid_steps", self.hybrid_steps)
        _require_non_negative_int("critic_iterations", self.critic_iterations)
        if self.critic_iterations == 0:
            raise ValueError("critic_iterations must be positive.")
        _require_non_negative_int("batch_size", self.batch_size)
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")
        _require_positive_number("learning_rate", self.learning_rate)
        if len(self.betas) != 2:
            raise ValueError("betas must contain two values.")
        for index, beta in enumerate(self.betas):
            require_finite_number(f"betas[{index}]", beta)
            if beta < 0.0 or beta >= 1.0:
                raise ValueError("betas must be between 0 inclusive and 1 exclusive.")
        object.__setattr__(self, "betas", tuple(map(float, self.betas)))
        _require_non_negative_number(
            "gradient_penalty_weight",
            self.gradient_penalty_weight,
        )
        _require_non_negative_int(
            "diffusion_reference_count",
            self.diffusion_reference_count,
        )
        if self.diffusion_reference_count == 0:
            raise ValueError("diffusion_reference_count must be positive.")
        require_finite_number(
            "diffusion_mix_probability",
            self.diffusion_mix_probability,
        )
        if self.diffusion_mix_probability < 0.0 or self.diffusion_mix_probability > 1.0:
            raise ValueError("diffusion_mix_probability must be between 0 and 1.")


@dataclass(frozen=True)
class SliceGANConditionConfig:
    steps: int = 800
    finetune_steps: int = 500
    noise_candidates: int = 8
    noise_lr: float = 5e-2
    noise_critic_weight: float = 1e-2
    finetune_generator_lr: float = 1e-5
    finetune_noise_lr: float = 2e-3
    finetune_critic_weight: float = 2e-2
    finetune_phase_weight: float = 50.0
    finetune_preservation_weight: float = 5.0
    anchor_transition_weight: float = 5.0
    anchor_influence_sigma: float = 8.0
    target_mismatch: float = 0.08

    def __post_init__(self) -> None:
        _require_non_negative_int("steps", self.steps)
        _require_non_negative_int("finetune_steps", self.finetune_steps)
        _require_non_negative_int("noise_candidates", self.noise_candidates)
        if self.noise_candidates == 0:
            raise ValueError("noise_candidates must be positive.")
        for name in (
            "noise_lr",
            "finetune_generator_lr",
            "finetune_noise_lr",
            "anchor_influence_sigma",
        ):
            _require_positive_number(name, getattr(self, name))
        for name in (
            "noise_critic_weight",
            "finetune_critic_weight",
            "finetune_phase_weight",
            "finetune_preservation_weight",
            "anchor_transition_weight",
        ):
            _require_non_negative_number(name, getattr(self, name))
        require_finite_number("target_mismatch", self.target_mismatch)
        if self.target_mismatch < 0.0 or self.target_mismatch > 1.0:
            raise ValueError("target_mismatch must be between 0 and 1.")


@dataclass(frozen=True)
class SliceGANRenderConfig:
    core_noise_size: int = 4
    halo_noise_size: int = 4

    def __post_init__(self) -> None:
        _require_non_negative_int("core_noise_size", self.core_noise_size)
        if self.core_noise_size == 0:
            raise ValueError("core_noise_size must be positive.")
        _require_non_negative_int("halo_noise_size", self.halo_noise_size)


@dataclass(frozen=True)
class SliceGANConfig:
    training: SliceGANTrainConfig = field(default_factory=SliceGANTrainConfig)
    conditioning: SliceGANConditionConfig = field(
        default_factory=SliceGANConditionConfig
    )
    rendering: SliceGANRenderConfig = field(default_factory=SliceGANRenderConfig)
    seed: int = 0
    intersection_tolerance: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.training, SliceGANTrainConfig):
            raise TypeError("training must be SliceGANTrainConfig.")
        if not isinstance(self.conditioning, SliceGANConditionConfig):
            raise TypeError("conditioning must be SliceGANConditionConfig.")
        if not isinstance(self.rendering, SliceGANRenderConfig):
            raise TypeError("rendering must be SliceGANRenderConfig.")
        _require_non_negative_int("seed", self.seed)
        require_finite_number(
            "intersection_tolerance",
            self.intersection_tolerance,
        )
        if self.intersection_tolerance < 0.0 or self.intersection_tolerance > 1.0:
            raise ValueError("intersection_tolerance must be between 0 and 1.")
