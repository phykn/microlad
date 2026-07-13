from collections.abc import Sequence

import torch

from src.validation import require_int


def _validate_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")


def _validate_axis(axis: int) -> None:
    require_int("axis", axis)

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")


def _validate_indices(volume: torch.Tensor, axis: int, indices: Sequence[int]) -> None:
    if len(indices) == 0:
        raise ValueError("indices must be non-empty.")

    for index in indices:
        require_int("index", index)

    if any(index < 0 or index >= volume.shape[axis] for index in indices):
        raise ValueError("indices must be inside the selected axis.")


def _validate_index(volume: torch.Tensor, axis: int, index: int) -> None:
    require_int("index", index)

    if index < 0 or index >= volume.shape[axis]:
        raise ValueError("index must be inside the selected axis.")


def _slice_shape(volume: torch.Tensor, axis: int) -> torch.Size:
    if axis == 0:
        return torch.Size([volume.shape[1], volume.shape[2]])

    if axis == 1:
        return torch.Size([volume.shape[0], volume.shape[2]])

    return torch.Size([volume.shape[0], volume.shape[1]])


def extract_slice(volume: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    _validate_volume(volume)
    _validate_axis(axis)
    _validate_index(volume, axis, index)

    if axis == 0:
        return volume[index, :, :]

    if axis == 1:
        return volume[:, index, :]

    return volume[:, :, index]


def replace_slice(
    volume: torch.Tensor,
    axis: int,
    index: int,
    image: torch.Tensor,
) -> None:
    _validate_volume(volume)
    _validate_axis(axis)
    _validate_index(volume, axis, index)

    if image.shape != _slice_shape(volume, axis):
        raise ValueError("image shape must match the selected slice shape.")
    if image.device != volume.device:
        raise ValueError("image device must match volume device.")
    if image.dtype != volume.dtype:
        raise ValueError("image dtype must match volume dtype.")

    if axis == 0:
        volume[index, :, :] = image
    elif axis == 1:
        volume[:, index, :] = image
    else:
        volume[:, :, index] = image


def extract_slice_batch(
    volume: torch.Tensor,
    axis: int,
    indices: Sequence[int],
) -> torch.Tensor:
    _validate_volume(volume)
    _validate_axis(axis)
    _validate_indices(volume, axis, indices)

    index_tensor = torch.as_tensor(indices, device=volume.device, dtype=torch.long)

    if axis == 0:
        return volume[index_tensor, :, :]

    if axis == 1:
        return volume[:, index_tensor, :].permute(1, 0, 2).contiguous()

    return volume[:, :, index_tensor].permute(2, 0, 1).contiguous()


def replace_slice_batch(
    volume: torch.Tensor,
    axis: int,
    indices: Sequence[int],
    images: torch.Tensor,
) -> None:
    _validate_volume(volume)
    _validate_axis(axis)
    _validate_indices(volume, axis, indices)

    if images.ndim != 3:
        raise ValueError("images must have shape [B, H, W].")

    if len(indices) != images.shape[0]:
        raise ValueError("indices length must match image batch size.")

    if images.shape[1:] != _slice_shape(volume, axis):
        raise ValueError("image shape must match the selected slice shape.")
    if images.device != volume.device:
        raise ValueError("images device must match volume device.")
    if images.dtype != volume.dtype:
        raise ValueError("images dtype must match volume dtype.")

    index_tensor = torch.as_tensor(indices, device=volume.device, dtype=torch.long)

    if axis == 0:
        volume[index_tensor, :, :] = images
    elif axis == 1:
        volume[:, index_tensor, :] = images.permute(1, 0, 2).contiguous()
    else:
        volume[:, :, index_tensor] = images.permute(1, 2, 0).contiguous()
