from dataclasses import dataclass

from src.common.validation import require_finite_number, require_int
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.modeling.phases.quantization import MAX_UINT8_PHASES


@dataclass(frozen=True)
class PredictOptions:
    num_phases: int

    anchor_segment: bool = False
    anchor_weight: float = 1.0
    anchor_fit_steps: int = 0
    anchor_fit_lr: float = 1e-1
    anchor_slab_radius: int = 2
    anchor_slab_weight: float = 0.25
    anchor_latent_sigma: float = 0.0
    anchor_latent_strength: float = 1.0
    lmpdd_axis_consensus: bool = False

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
    sds_balanced_slices: bool = False
    sds_consensus: bool = True
    sds_lr: float = 1e-2
    sds_t_min: int = 1
    sds_t_max: int | None = None
    sds_weight: float = 1.0

    joint_3d_steps: int = 0
    joint_3d_batch_size: int = 8
    joint_3d_lr: float = 1e-4
    joint_3d_entropy_weight: float = 1e-2
    joint_3d_continuity_weight: float = 5e-2
    joint_3d_transition_weight: float = 0.0
    joint_3d_run_weight: float = 0.0
    joint_3d_patch_weight: float = 0.0
    joint_3d_texture_weight: float = 0.0
    joint_3d_interface_weight: float = 0.0
    joint_3d_discriminator_lr: float = 1e-4

    slicegan_steps: int = 0
    slicegan_hybrid_steps: int = 1000
    slicegan_condition_steps: int = 800
    slicegan_finetune_steps: int = 500
    slicegan_seed: int = 0

    refine_steps: int = 0

    def __post_init__(self) -> None:
        require_int("num_phases", self.num_phases)

        if self.num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        if self.num_phases > MAX_UINT8_PHASES:
            raise ValueError(
                f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 output."
            )

        for name, weight in (
            ("anchor_weight", self.anchor_weight),
            ("anchor_slab_weight", self.anchor_slab_weight),
            ("vf_weight", self.vf_weight),
            ("tpc_weight", self.tpc_weight),
            ("sa_weight", self.sa_weight),
            ("diffusivity_weight", self.diffusivity_weight),
        ):
            require_finite_number(name, weight)

            if weight < 0.0 or weight > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        require_finite_number("diffusivity_low_cond", self.diffusivity_low_cond)

        if self.diffusivity_low_cond < 0.0 or self.diffusivity_low_cond > 1.0:
            raise ValueError("diffusivity_low_cond must be between 0 and 1.")

        require_int("anchor_fit_steps", self.anchor_fit_steps)
        if self.anchor_fit_steps < 0:
            raise ValueError("anchor_fit_steps must be non-negative.")

        require_finite_number("anchor_fit_lr", self.anchor_fit_lr)
        if self.anchor_fit_lr <= 0.0:
            raise ValueError("anchor_fit_lr must be positive.")

        require_int("anchor_slab_radius", self.anchor_slab_radius)
        if self.anchor_slab_radius < 0:
            raise ValueError("anchor_slab_radius must be non-negative.")

        require_finite_number("anchor_latent_sigma", self.anchor_latent_sigma)
        if self.anchor_latent_sigma < 0.0:
            raise ValueError("anchor_latent_sigma must be non-negative.")

        require_finite_number("anchor_latent_strength", self.anchor_latent_strength)
        if self.anchor_latent_strength <= 0.0 or self.anchor_latent_strength > 1.0:
            raise ValueError(
                "anchor_latent_strength must be greater than 0 and at most 1."
            )

        if not isinstance(self.lmpdd_axis_consensus, bool):
            raise ValueError("lmpdd_axis_consensus must be a boolean.")

        require_int("sds_steps", self.sds_steps)
        if self.sds_steps < 0:
            raise ValueError("sds_steps must be non-negative.")

        require_int("sds_slice_steps", self.sds_slice_steps)
        if self.sds_slice_steps < 0:
            raise ValueError("sds_slice_steps must be non-negative.")

        require_int("sds_batch_size", self.sds_batch_size)
        if self.sds_batch_size <= 0:
            raise ValueError("sds_batch_size must be positive.")

        if not isinstance(self.sds_balanced_slices, bool):
            raise ValueError("sds_balanced_slices must be a boolean.")

        if not isinstance(self.sds_consensus, bool):
            raise ValueError("sds_consensus must be a boolean.")

        require_finite_number("sds_lr", self.sds_lr)
        if self.sds_lr <= 0.0:
            raise ValueError("sds_lr must be positive.")

        require_int("sds_t_min", self.sds_t_min)
        if self.sds_t_min < 0:
            raise ValueError("sds_t_min must be non-negative.")

        if self.sds_t_max is not None:
            require_int("sds_t_max", self.sds_t_max)

            if self.sds_t_max <= self.sds_t_min:
                raise ValueError("sds_t_max must be greater than sds_t_min.")

        require_finite_number("sds_weight", self.sds_weight)
        if self.sds_weight < 0.0 or self.sds_weight > 1.0:
            raise ValueError("sds_weight must be between 0 and 1.")

        require_int("joint_3d_steps", self.joint_3d_steps)
        if self.joint_3d_steps < 0:
            raise ValueError("joint_3d_steps must be non-negative.")

        require_int("joint_3d_batch_size", self.joint_3d_batch_size)
        if self.joint_3d_batch_size <= 0:
            raise ValueError("joint_3d_batch_size must be positive.")

        require_finite_number("joint_3d_lr", self.joint_3d_lr)
        if self.joint_3d_lr <= 0.0:
            raise ValueError("joint_3d_lr must be positive.")

        for name, weight in (
            ("joint_3d_entropy_weight", self.joint_3d_entropy_weight),
            ("joint_3d_continuity_weight", self.joint_3d_continuity_weight),
            ("joint_3d_transition_weight", self.joint_3d_transition_weight),
            ("joint_3d_run_weight", self.joint_3d_run_weight),
            ("joint_3d_patch_weight", self.joint_3d_patch_weight),
            ("joint_3d_texture_weight", self.joint_3d_texture_weight),
            ("joint_3d_interface_weight", self.joint_3d_interface_weight),
        ):
            require_finite_number(name, weight)
            if weight < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        require_finite_number(
            "joint_3d_discriminator_lr",
            self.joint_3d_discriminator_lr,
        )
        if self.joint_3d_discriminator_lr <= 0.0:
            raise ValueError("joint_3d_discriminator_lr must be positive.")

        for name, value in (
            ("slicegan_steps", self.slicegan_steps),
            ("slicegan_hybrid_steps", self.slicegan_hybrid_steps),
            ("slicegan_condition_steps", self.slicegan_condition_steps),
            ("slicegan_finetune_steps", self.slicegan_finetune_steps),
            ("slicegan_seed", self.slicegan_seed),
        ):
            require_int(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

        require_int("refine_steps", self.refine_steps)
        if self.refine_steps < 0:
            raise ValueError("refine_steps must be non-negative.")

        if self.joint_3d_steps > 0 and self.sds_steps > 0:
            raise ValueError("joint_3d_steps and sds_steps cannot both be positive.")
        if self.joint_3d_steps > 0 and self.refine_steps > 0:
            raise ValueError("joint 3D optimization replaces refine_steps.")
        if self.joint_3d_steps > 0 and self.anchor_fit_steps > 0:
            raise ValueError("joint 3D optimization replaces anchor_fit_steps.")
        if self.slicegan_steps > 0 and (
            self.sds_steps > 0
            or self.joint_3d_steps > 0
            or self.refine_steps > 0
            or self.anchor_fit_steps > 0
        ):
            raise ValueError(
                "conditional SliceGAN replaces sds_steps, joint_3d_steps, "
                "refine_steps, and anchor_fit_steps."
            )
