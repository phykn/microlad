from collections.abc import Mapping, Sequence

import torch

from src.guidance.conditioning.model import AnchorSlice
from src.guidance.physics.diffusivity import DiffusivitySolver
from src.reconstruction.slices import extract_slice

def _validate_inputs(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    axis: int,
    index: int,
    steps: int,
    lr: float,
    sds_weight: float,
    anchor_weight: float,
    anchor_target: torch.Tensor | None,
    vf_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    tpc_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    sa_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if index < 0 or index >= volume.shape[axis]:
        raise ValueError("index must be inside the selected axis.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    _validate_optimization_contract(
        lr=lr,
        sds_weight=sds_weight,
        anchor_weight=anchor_weight,
        anchor_target=anchor_target,
        vf_weight=vf_weight,
        vf_targets=vf_targets,
        tpc_weight=tpc_weight,
        tpc_targets=tpc_targets,
        sa_weight=sa_weight,
        sa_targets=sa_targets,
        diffusivity_weight=diffusivity_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
    )

    image_size = int(vae.image_size)
    if extract_slice(volume, axis, index).shape != torch.Size(
        [image_size, image_size]
    ):
        raise ValueError("selected slice shape must match vae.image_size.")


def _validate_optimization_contract(
    *,
    lr: float,
    sds_weight: float,
    anchor_weight: float,
    anchor_target: torch.Tensor | None,
    require_anchor_target: bool = True,
    vf_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    tpc_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    sa_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
) -> None:
    if lr <= 0.0:
        raise ValueError("lr must be positive.")

    for name, weight in (
        ("sds_weight", sds_weight),
        ("anchor_weight", anchor_weight),
        ("vf_weight", vf_weight),
        ("tpc_weight", tpc_weight),
        ("sa_weight", sa_weight),
        ("diffusivity_weight", diffusivity_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    if require_anchor_target and anchor_weight > 0.0 and anchor_target is None:
        raise ValueError("anchor_target is required when anchor_weight is positive.")

    if vf_weight > 0.0 and vf_targets is None:
        raise ValueError("vf_targets is required when vf_weight is positive.")

    if tpc_weight > 0.0 and tpc_targets is None:
        raise ValueError("tpc_targets is required when tpc_weight is positive.")

    if sa_weight > 0.0 and sa_targets is None:
        raise ValueError("sa_targets is required when sa_weight is positive.")

    if diffusivity_weight > 0.0 and diffusivity_targets is None:
        raise ValueError(
            "diffusivity_targets is required when diffusivity_weight is positive."
        )

    if diffusivity_weight > 0.0:
        if diffusivity_solver is None:
            raise ValueError("diffusivity_solver is required for diffusivity loss.")


def _validate_volume_inputs(
    volume: torch.Tensor,
    *,
    steps: int,
    slice_steps: int,
    sds_batch_size: int,
    slice_schedule: Sequence[tuple[int, int]] | None,
    anchors: Sequence[AnchorSlice] | None,
    anchor_weight: float,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    if any(size <= 0 for size in volume.shape):
        raise ValueError("volume dimensions must be positive.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    if slice_steps < 0:
        raise ValueError("slice_steps must be non-negative.")

    if not isinstance(sds_batch_size, int) or isinstance(sds_batch_size, bool):
        raise ValueError("sds_batch_size must be an integer.")

    if sds_batch_size <= 0:
        raise ValueError("sds_batch_size must be positive.")

    if slice_schedule is not None and len(slice_schedule) < steps * sds_batch_size:
        raise ValueError("slice_schedule must contain one entry per batched slice.")

    if anchor_weight > 0.0 and not anchors:
        raise ValueError("anchors are required when anchor_weight is positive.")
