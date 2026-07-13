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
    if volume_size < image_size:
        raise ValueError("volume_size must be at least vae.image_size.")

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
    base_size = volume_size == image_size
    active_steps = options.joint.steps if base_size else options.scale.steps
    if base_size and anchors and active_steps <= 0:
        raise ValueError("base-size anchors require joint.steps to be positive.")
    has_guidance = descriptor_targets or options.phase_fractions is not None
    if has_guidance and active_steps <= 0:
        raise ValueError(
            "phase fractions require active guidance steps."
        )
    if descriptor_targets and target_labels is None:
        raise ValueError("target_images are required for descriptor target losses.")
    if base_size and options.critic.steps > 0 and target_labels is None and not anchors:
        raise ValueError("critic training requires target_images or anchors.")
    target_consumer = descriptor_targets or (
        base_size and options.critic.steps > 0
    )
    if target_labels is not None and not target_consumer:
        raise ValueError("target_images require an enabled target or critic setting.")
    t_max: int | None = None
    if active_steps > 0:
        t_max = resolve_t_max(options, num_timesteps)

    descriptor_tile_size: int | None = None
    if target_labels is not None:
        height, width = map(int, target_labels.shape[-2:])
        if height != width:
            raise ValueError("target images must be square.")
        target_size = height
        if base_size and target_size != image_size:
            raise ValueError("target images must match vae.image_size.")
        if not base_size:
            if target_size == image_size:
                descriptor_tile_size = target_size if descriptor_targets else None
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


def uses_descriptor_targets(options: PredictOptions) -> bool:
    return (
        (
            (
                options.targets.slice_fraction_weight > 0.0
                or options.targets.global_fraction_weight > 0.0
            )
            and options.phase_fractions is None
        )
        or options.targets.tpc_weight > 0.0
        or options.targets.surface_area_weight > 0.0
        or options.targets.diffusivity_weight > 0.0
    )
