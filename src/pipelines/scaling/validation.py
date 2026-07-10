import math
from collections.abc import Mapping, Sequence

import torch

from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.common.tensors.validation import require_finite, require_float

def _validate_inputs(
    volume: torch.Tensor,
    *,
    steps: int,
    slice_steps: int,
    sds_batch_size: int,
    lr: float,
    slice_schedule: Sequence[tuple[int, int]] | None,
    anchors: Sequence[AnchorSlice] | None,
    anchor_targets: Mapping[tuple[int, int], torch.Tensor] | None,
    anchor_masks: Mapping[tuple[int, int], torch.Tensor] | None,
    anchor_weight: float,
    sds_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    vf_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
    diffusivity_weight: float,
    temperature: float,
    num_phases: int,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    require_float("volume dtype", volume.dtype)
    require_finite("volume", volume)

    depth, height, width = volume.shape
    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("scale-up SDS requires a cubic volume.")

    _validate_non_negative_integer("steps", steps)
    _validate_non_negative_integer("slice_steps", slice_steps)

    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("num_phases must be an integer.")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    _validate_positive_scalar("lr", lr)
    _validate_positive_scalar("temperature", temperature)

    if not isinstance(sds_batch_size, int) or isinstance(sds_batch_size, bool):
        raise ValueError("sds_batch_size must be an integer.")

    if sds_batch_size <= 0:
        raise ValueError("sds_batch_size must be positive.")

    if slice_schedule is not None and len(slice_schedule) < steps * sds_batch_size:
        raise ValueError("slice_schedule must contain one entry per batched slice.")

    for name, weight in (
        ("sds_weight", sds_weight),
        ("anchor_weight", anchor_weight),
        ("vf_weight", vf_weight),
        ("tpc_weight", tpc_weight),
        ("sa_weight", sa_weight),
        ("diffusivity_weight", diffusivity_weight),
    ):
        _validate_non_negative_scalar(name, weight)

    _validate_anchor_tensor_map(
        "anchor_targets",
        anchor_targets,
        volume_shape=volume.shape,
    )
    _validate_anchor_tensor_map(
        "anchor_masks",
        anchor_masks,
        volume_shape=volume.shape,
        mask=True,
    )

    if anchor_weight > 0.0 and not anchors and not anchor_targets:
        raise ValueError("anchors are required when anchor_weight is positive.")

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

    if diffusivity_weight > 0.0 and diffusivity_solver is None:
        raise ValueError("diffusivity_solver is required for diffusivity loss.")


def _as_anchor_image(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4 and target.shape[:2] == (1, 1):
        return target[0, 0]

    if target.ndim == 2:
        return target

    raise ValueError("anchor target must have shape [H, W] or [1, 1, H, W].")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


def _validate_non_negative_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")


def _validate_anchor_tensor_map(
    name: str,
    values: Mapping[tuple[int, int], torch.Tensor] | None,
    *,
    volume_shape: torch.Size,
    mask: bool = False,
) -> None:
    if not values:
        return

    for key, value in values.items():
        axis, index = _validate_anchor_tensor_key(
            name,
            key,
            volume_shape=volume_shape,
        )
        image = _as_anchor_image(value)
        if image.shape != _slice_shape(volume_shape, axis):
            raise ValueError(f"{name}[{axis}, {index}] shape must match selected slice.")

        require_finite(f"{name}[{axis}, {index}]", image)

        if mask and (image.min().item() < 0.0 or image.max().item() > 1.0):
            raise ValueError(f"{name} values must be between 0 and 1.")


def _validate_anchor_tensor_key(
    name: str,
    key: tuple[int, int],
    *,
    volume_shape: torch.Size,
) -> tuple[int, int]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(f"{name} keys must be (axis, index).")

    axis, index = key
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or not isinstance(index, int)
        or isinstance(index, bool)
    ):
        raise ValueError(f"{name} keys must contain integer axis and index.")

    if axis not in (0, 1, 2):
        raise ValueError(f"{name} axis must be 0, 1, or 2.")

    if index < 0 or index >= volume_shape[axis]:
        raise ValueError(f"{name} index must be inside the selected axis.")

    return axis, index


def _slice_shape(volume_shape: torch.Size, axis: int) -> torch.Size:
    if axis == 0:
        return torch.Size([volume_shape[1], volume_shape[2]])

    if axis == 1:
        return torch.Size([volume_shape[0], volume_shape[2]])

    return torch.Size([volume_shape[0], volume_shape[1]])


def _tensor_map(
    values: Mapping[tuple[int, int], torch.Tensor] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[tuple[int, int], torch.Tensor]:
    if not values:
        return {}

    return {
        (int(axis), int(index)): value.to(device=device, dtype=dtype)
        for (axis, index), value in values.items()
    }
