from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import tile_starts


@dataclass
class ParsedConditionImage:
    image: np.ndarray | torch.Tensor
    axis: int
    slice_index: int


@dataclass
class ConditionOptions:
    size: int
    images: list[ParsedConditionImage]
    condition_weight: float
    stats_weight: float


@dataclass
class ConditionLock:
    condition_z: torch.Tensor
    axis: int
    slice_index: int
    row_start: int = 0
    col_start: int = 0
    condition_index: int = 0


@dataclass
class FixedSlice:
    axis: int
    index: int
    image: torch.Tensor


@dataclass
class EncodedConditions:
    locks: list[ConditionLock]
    fixed_slices: list[FixedSlice]
    condition_slices: list[FixedSlice]
    condition_images: list[torch.Tensor]


def parse_condition_options(
    condition: dict[str, object] | None,
    default_size: int,
    downsample: int,
) -> ConditionOptions:
    if condition is None:
        return ConditionOptions(size=default_size, images=[], condition_weight=0.0, stats_weight=0.0)
    if not isinstance(condition, dict):
        raise ValueError("condition must be a dict or None.")

    size = int(condition.get("size", default_size))
    if size <= 0 or size % downsample != 0:
        raise ValueError("condition size must be positive and divisible by downsample.")

    condition_weight = float(condition.get("condition_weight", 0.0))
    if condition_weight < 0:
        raise ValueError("condition_weight must be non-negative.")

    stats_weight = float(condition.get("stats_weight", 0.0))
    if stats_weight < 0:
        raise ValueError("stats_weight must be non-negative.")

    images = condition.get("images", [])
    if images is None:
        images = []
    if not isinstance(images, list):
        raise ValueError("condition images must be a list.")

    specs = []
    for item in images:
        if not isinstance(item, dict):
            raise ValueError("each condition image must be a dict.")
        if "image" not in item:
            raise ValueError("each condition image must include image.")
        index = int(item["index"])
        if index < 0 or index >= size:
            raise ValueError("condition image index must be inside size.")
        specs.append(
            ParsedConditionImage(
                image=item["image"],
                axis=int(item["axis"]),
                slice_index=index,
            )
        )
    return ConditionOptions(size=size, images=specs, condition_weight=condition_weight, stats_weight=stats_weight)


def condition_to_image(
    condition: np.ndarray | torch.Tensor,
    device: torch.device,
    size: int | None = None,
) -> torch.Tensor:
    if isinstance(condition, np.ndarray):
        condition_image = torch.from_numpy(condition)
    elif isinstance(condition, torch.Tensor):
        condition_image = condition
    else:
        raise ValueError("condition must be a numpy array or torch tensor.")

    condition_image = condition_image.to(device=device, dtype=torch.float32)
    if condition_image.ndim == 2:
        condition_image = condition_image.unsqueeze(0).unsqueeze(0)
    elif condition_image.ndim == 3:
        condition_image = condition_image.unsqueeze(0)
    if condition_image.ndim != 4 or condition_image.shape[0] != 1 or condition_image.shape[1] != 1:
        raise ValueError("condition must have shape [H, W], [1, H, W], or [1, 1, H, W].")
    if float(condition_image.max()) > 1.0:
        condition_image = condition_image / 255.0
    if size is not None:
        if size <= 0:
            raise ValueError("size must be positive.")
        if condition_image.shape[-2:] != (size, size):
            condition_image = F.interpolate(condition_image, size=(size, size), mode="bilinear", align_corners=False)
    return condition_image


def encode_condition(
    vae: torch.nn.Module,
    condition: np.ndarray | torch.Tensor,
    device: torch.device,
    image_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    condition_image = condition_to_image(condition, device, size=image_size)
    condition_z, _ = vae.encode(condition_image * 2 - 1)
    return condition_z.squeeze(0), condition_image


def encode_condition_items(
    vae: torch.nn.Module,
    conditions: list[ParsedConditionImage],
    lock_condition_slice: bool,
    device: torch.device,
    image_size: int | None = None,
) -> EncodedConditions:
    locks = []
    fixed_slices = []
    condition_slices = []
    condition_images = []
    for condition_index, item in enumerate(conditions):
        condition_z, condition_image = encode_condition(
            vae=vae,
            condition=item.image,
            device=device,
            image_size=image_size,
        )
        if condition_image is not None:
            condition_images.append(condition_image)
            condition_slices.append(FixedSlice(axis=int(item.axis), index=int(item.slice_index), image=condition_image.squeeze(0)))
            if lock_condition_slice:
                fixed_slices.append(
                    FixedSlice(axis=int(item.axis), index=int(item.slice_index), image=condition_image.squeeze(0))
                )

        locks.append(
            ConditionLock(
                condition_z=condition_z,
                axis=int(item.axis),
                slice_index=int(item.slice_index),
                condition_index=condition_index,
            )
        )

    return EncodedConditions(
        locks=locks,
        fixed_slices=fixed_slices,
        condition_slices=condition_slices,
        condition_images=condition_images,
    )


def encode_tiled_condition_items(
    vae: torch.nn.Module,
    conditions: list[ParsedConditionImage],
    tile_size: int,
    tile_overlap: int,
    downsample: int,
    lock_condition_slice: bool,
    device: torch.device,
    image_size: int | None = None,
) -> EncodedConditions:
    if not conditions:
        raise ValueError("conditions must not be empty.")
    if tile_size <= 0 or tile_size % downsample != 0:
        raise ValueError("tile_size must be positive and divisible by downsample.")
    if tile_overlap < 0 or tile_overlap >= tile_size or tile_overlap % downsample != 0:
        raise ValueError("tile_overlap must be non-negative, smaller than tile_size, and divisible by downsample.")

    locks = []
    fixed_slices = []
    condition_slices = []
    condition_images = []
    for condition_index, item in enumerate(conditions):
        axis = int(item.axis)
        slice_index = int(item.slice_index)
        condition_image = condition_to_image(item.image, device, size=image_size)
        height, width = int(condition_image.shape[-2]), int(condition_image.shape[-1])
        if height < tile_size or width < tile_size:
            raise ValueError("condition image must be at least tile_size in both dimensions.")

        condition_images.append(condition_image)
        condition_slices.append(FixedSlice(axis=axis, index=slice_index, image=condition_image.squeeze(0)))
        if lock_condition_slice:
            fixed_slices.append(FixedSlice(axis=axis, index=slice_index, image=condition_image.squeeze(0)))

        row_starts = tile_starts(height, tile_size, tile_overlap)
        col_starts = tile_starts(width, tile_size, tile_overlap)
        for row_start in row_starts:
            for col_start in col_starts:
                tile = condition_image[:, :, row_start:row_start + tile_size, col_start:col_start + tile_size]
                condition_z, _ = vae.encode(tile * 2 - 1)
                locks.append(
                    ConditionLock(
                        condition_z=condition_z.squeeze(0),
                        axis=axis,
                        slice_index=slice_index,
                        row_start=row_start // downsample,
                        col_start=col_start // downsample,
                        condition_index=condition_index,
                    )
                )

    return EncodedConditions(
        locks=locks,
        fixed_slices=fixed_slices,
        condition_slices=condition_slices,
        condition_images=condition_images,
    )
