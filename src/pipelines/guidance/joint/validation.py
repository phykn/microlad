from collections.abc import Sequence

import torch

from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.validation import require_finite


def validate_inputs(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    num_phases: int,
    anchors: Sequence[AnchorSlice] | None,
    anchor_weight: float,
    anchor_slab_radius: int,
    anchor_slab_weight: float,
    entropy_weight: float,
    continuity_weight: float,
    transition_weight: float,
    run_weight: float,
    patch_weight: float,
    texture_weight: float,
    interface_weight: float,
    discriminator_lr: float,
) -> None:
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("joint 3D optimization requires a cubic [D, H, W] volume.")
    require_finite("volume", volume)
    if volume.is_floating_point() and not torch.equal(volume, volume.round()):
        raise ValueError("joint 3D volume must contain integer phase values.")
    if volume.min().item() < 0 or volume.max().item() >= num_phases:
        raise ValueError("joint 3D volume labels must be inside the phase range.")
    if int(volume.shape[0]) != int(vae.image_size):
        raise ValueError("joint 3D volume size must match vae.image_size.")
    if getattr(vae, "num_phases", None) != num_phases:
        raise ValueError("joint 3D optimization requires a matching categorical VAE.")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer.")
    if batch_size <= 0 or batch_size > int(volume.shape[0]):
        raise ValueError("batch_size must be between 1 and the volume size.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    for name, weight in (
        ("anchor_weight", anchor_weight),
        ("entropy_weight", entropy_weight),
        ("continuity_weight", continuity_weight),
        ("transition_weight", transition_weight),
        ("run_weight", run_weight),
        ("patch_weight", patch_weight),
        ("texture_weight", texture_weight),
        ("interface_weight", interface_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not isinstance(anchor_slab_radius, int) or isinstance(anchor_slab_radius, bool):
        raise ValueError("anchor_slab_radius must be an integer.")
    if anchor_slab_radius < 0:
        raise ValueError("anchor_slab_radius must be non-negative.")
    if anchor_slab_weight < 0.0 or anchor_slab_weight > 1.0:
        raise ValueError("anchor_slab_weight must be between 0 and 1.")
    if discriminator_lr <= 0.0:
        raise ValueError("discriminator_lr must be positive.")
    if anchor_weight > 0.0 and not anchors:
        raise ValueError("anchors are required when anchor_weight is positive.")
