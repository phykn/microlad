from dataclasses import dataclass

import torch

from .conditioned_sampling import voxel_to_latent_index


@dataclass
class ConditionSpec:
    condition: torch.Tensor
    axis: int
    slice_index: int


@dataclass
class EncodedConditions:
    condition_slices: list[dict[str, torch.Tensor | int]]
    fixed_slices: list[dict[str, torch.Tensor | int]]
    first_condition_image: torch.Tensor | None


def condition_specs_to_dicts(conditions: list[ConditionSpec]) -> list[dict[str, torch.Tensor | int]]:
    return [
        {"condition": item.condition, "axis": item.axis, "slice_index": item.slice_index}
        for item in conditions
    ]


def encode_condition(
    vae: torch.nn.Module,
    condition: torch.Tensor,
    condition_is_latent: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if condition_is_latent:
        condition_z = condition.to(device)
        if condition_z.ndim == 4:
            condition_z = condition_z.squeeze(0)
        return condition_z, None

    condition_image = condition.to(device)
    if condition_image.ndim == 3:
        condition_image = condition_image.unsqueeze(0)
    condition_z, _ = vae.encode(condition_image * 2 - 1)
    return condition_z.squeeze(0), condition_image


def encode_condition_items(
    vae: torch.nn.Module,
    conditions: list[dict[str, torch.Tensor | int]],
    condition_is_latent: bool,
    lock_condition_slice: bool,
    device: torch.device,
) -> EncodedConditions:
    if not conditions:
        raise ValueError("conditions must not be empty.")

    condition_slices = []
    fixed_slices = []
    first_condition_image = None
    for item in conditions:
        axis = int(item["axis"])
        slice_index = int(item["slice_index"])
        condition = item["condition"]
        if not isinstance(condition, torch.Tensor):
            raise ValueError("condition item must include tensor condition.")

        condition_z, condition_image = encode_condition(
            vae=vae,
            condition=condition,
            condition_is_latent=condition_is_latent,
            device=device,
        )
        if condition_image is not None:
            if first_condition_image is None:
                first_condition_image = condition_image
            if lock_condition_slice:
                fixed_slices.append({"axis": axis, "index": slice_index, "image": condition_image.squeeze(0)})

        condition_slices.append({"condition_z": condition_z, "axis": axis, "slice_index": slice_index})

    return EncodedConditions(
        condition_slices=condition_slices,
        fixed_slices=fixed_slices,
        first_condition_image=first_condition_image,
    )


def condition_error_from_volume(
    volume_z: torch.Tensor,
    condition_z: torch.Tensor,
    axis: int,
    slice_index: int,
) -> float:
    latent_index = voxel_to_latent_index(slice_index)
    if axis == 0:
        fixed = volume_z[:, latent_index, :, :]
    elif axis == 1:
        fixed = volume_z[:, :, latent_index, :]
    else:
        fixed = volume_z[:, :, :, latent_index]
    return float((fixed - condition_z).abs().max())


def _condition_hw(condition: torch.Tensor) -> tuple[int, int]:
    if condition.ndim == 4:
        return int(condition.shape[2]), int(condition.shape[3])
    if condition.ndim == 3:
        return int(condition.shape[1]), int(condition.shape[2])
    raise ValueError("condition must have shape [C, H, W] or [B, C, H, W].")


def infer_scale_up_size(conditions: list[ConditionSpec], output_size: int | None, downsample: int) -> int:
    if not conditions:
        raise ValueError("conditions must not be empty.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")

    h, w = _condition_hw(conditions[0].condition)
    if h != w:
        raise ValueError("scale-up condition crop must be square.")

    size = int(output_size) if output_size is not None else h
    if size <= 0 or size % downsample != 0:
        raise ValueError("output_size must be positive and divisible by downsample.")
    if h != size or w != size:
        raise ValueError("condition crop size must match output_size.")

    for item in conditions:
        current_h, current_w = _condition_hw(item.condition)
        if current_h != size or current_w != size:
            raise ValueError("all scale-up conditions must match output_size.")
        if item.slice_index < 0 or item.slice_index >= size:
            raise ValueError("condition slice_index must be inside output_size.")
    return size


def scale_up_volume_shape(
    conditions: list[ConditionSpec],
    output_size: int | None,
    latent_ch: int,
    downsample: int,
) -> tuple[int, int, int, int]:
    size = infer_scale_up_size(conditions, output_size=output_size, downsample=downsample)
    latent_size = size // downsample
    return latent_ch, latent_size, latent_size, latent_size
