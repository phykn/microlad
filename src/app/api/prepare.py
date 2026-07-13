from collections.abc import Sequence

import numpy as np
import torch

from src.app.api.options import PredictOptions
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.conditioning.validation import (
    validate_anchor_positions,
    validate_anchors,
)


def anchor_size(anchors: Sequence[AnchorSlice] | None) -> int | None:
    if not anchors:
        return None

    size: int | None = None
    for anchor in anchors:
        if not isinstance(anchor.image, np.ndarray):
            raise TypeError("anchor image must be a numpy array.")
        if anchor.image.ndim != 2:
            raise ValueError("anchor image must be 2D.")

        height, width = anchor.image.shape
        if height != width:
            raise ValueError("anchor image must be square.")
        if size is None:
            size = int(height)
        elif size != int(height):
            raise ValueError("anchor images must have the same size.")

    return size


def prepare_prediction(
    options: PredictOptions,
    anchors: Sequence[AnchorSlice] | None,
    target_labels: torch.Tensor | None,
    volume_size: int | None,
    image_size: int,
    num_timesteps: int,
) -> tuple[int, int | None, int | None]:
    """Validates prediction inputs and resolves derived sizes.

    Args:
        options: Generation settings.
        anchors: Optional conditional slices.
        target_labels: Optional prepared categorical reference images.
        volume_size: Requested output size, or None to infer it.
        image_size: Spatial size used to train the VAE.
        num_timesteps: Number of diffusion timesteps.

    Returns:
        Resolved volume size, optional descriptor tile size, and optional
        maximum diffusion timestep.
    """

    condition_size = anchor_size(anchors)
    if volume_size is None:
        volume_size = condition_size or image_size
    elif not isinstance(volume_size, int) or isinstance(volume_size, bool):
        raise ValueError("volume_size must be an integer.")
    elif volume_size <= 0:
        raise ValueError("volume_size must be positive.")
    elif condition_size is not None and condition_size not in (
        image_size,
        volume_size,
    ):
        raise ValueError("anchor image size must match vae.image_size or volume_size.")

    if (
        options.slicegan is not None
        and condition_size is not None
        and condition_size != image_size
    ):
        raise ValueError(
            "conditional SliceGAN anchor images must match vae.image_size."
        )

    if volume_size > image_size and anchors:
        validate_anchor_positions(
            anchors or [],
            (volume_size, volume_size, volume_size),
        )
    else:
        validation_size = image_size if condition_size == image_size else volume_size
        if anchors:
            validate_anchors(
                anchors,
                (validation_size, validation_size, validation_size),
            )

    descriptor_targets = uses_descriptor_targets(options)
    joint_targets = uses_joint_targets(options)
    target_losses = descriptor_targets or joint_targets
    if options.slicegan is not None and target_losses:
        raise ValueError(
            "descriptor target losses are not used by conditional SliceGAN."
        )
    if joint_targets and options.joint.steps <= 0:
        raise ValueError("joint reference weights require joint.steps to be positive.")
    has_guidance = target_losses or options.phase_fractions is not None
    if (
        has_guidance
        and options.slicegan is None
        and options.sds.steps <= 0
        and options.joint.steps <= 0
    ):
        raise ValueError(
            "phase fractions and target losses require sds.steps or "
            "joint.steps to be positive."
        )
    if target_losses and target_labels is None:
        raise ValueError("target_images are required when target losses are enabled.")
    t_max: int | None = None
    if options.sds.steps > 0 or options.joint.steps > 0:
        t_max = resolve_t_max(options, num_timesteps)

    descriptor_tile_size: int | None = None
    if target_losses and options.sds.steps > 0:
        assert target_labels is not None
        height, width = map(int, target_labels.shape[-2:])
        if height != width:
            raise ValueError("scale-up target images must be square.")
        target_size = height
        if volume_size == image_size and target_size != image_size:
            raise ValueError("target images must match vae.image_size.")
        if volume_size != image_size:
            if target_size == image_size:
                descriptor_tile_size = target_size
            elif target_size != volume_size:
                raise ValueError(
                    "scale-up target images must match vae.image_size or volume_size."
                )

    return volume_size, descriptor_tile_size, t_max


def resolve_t_max(options: PredictOptions, num_timesteps: int) -> int:
    t_max = num_timesteps if options.prior.t_max is None else int(options.prior.t_max)
    if t_max <= options.prior.t_min:
        raise ValueError("prior.t_max must be greater than prior.t_min.")
    if t_max > num_timesteps:
        raise ValueError("prior.t_max must be at most the DDPMProcess schedule length.")
    return t_max


def uses_image_targets(options: PredictOptions) -> bool:
    return uses_descriptor_targets(options) or uses_joint_targets(options)


def uses_descriptor_targets(options: PredictOptions) -> bool:
    return (
        (options.targets.vf_weight > 0.0 and options.phase_fractions is None)
        or options.targets.tpc_weight > 0.0
        or options.targets.surface_area_weight > 0.0
        or options.targets.diffusivity_weight > 0.0
    )


def uses_joint_targets(options: PredictOptions) -> bool:
    return any(
        weight > 0.0
        for weight in (
            options.joint.transition_weight,
            options.joint.run_weight,
            options.joint.patch_weight,
            options.joint.texture_weight,
            options.joint.interface_weight,
        )
    )
