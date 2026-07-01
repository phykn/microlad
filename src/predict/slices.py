from collections.abc import Sequence

import torch


def _validate_axis(axis: int) -> None:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")


def _validate_indices(volume: torch.Tensor, axis: int, indices: Sequence[int]) -> None:
    if not indices:
        raise ValueError("indices must be non-empty.")
    if any(index < 0 or index >= volume.shape[axis] for index in indices):
        raise ValueError("indices must be inside the selected axis.")


def extract_slice(volume: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    _validate_axis(axis)
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
    _validate_axis(axis)
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
    _validate_axis(axis)
    _validate_indices(volume, axis, indices)
    if images.ndim != 3:
        raise ValueError("images must have shape [B, H, W].")
    if len(indices) != images.shape[0]:
        raise ValueError("indices length must match image batch size.")
    index_tensor = torch.as_tensor(indices, device=volume.device, dtype=torch.long)
    if axis == 0:
        volume[index_tensor, :, :] = images
    elif axis == 1:
        volume[:, index_tensor, :] = images.permute(1, 0, 2).contiguous()
    else:
        volume[:, :, index_tensor] = images.permute(1, 2, 0).contiguous()


def select_slice(
    volume: torch.Tensor,
    step: int,
    slice_schedule: Sequence[tuple[int, int]] | None,
) -> tuple[int, int]:
    if slice_schedule is None:
        axis = int(torch.randint(0, 3, (), device=volume.device).item())
        index = int(torch.randint(0, volume.shape[axis], (), device=volume.device).item())
    else:
        axis, index = slice_schedule[step]
        axis = int(axis)
        index = int(index)

    if axis not in (0, 1, 2):
        raise ValueError("slice_schedule axis must be 0, 1, or 2.")
    if index < 0 or index >= volume.shape[axis]:
        raise ValueError("slice_schedule index must be inside the selected axis.")
    return axis, index


def select_slice_batch(
    volume: torch.Tensor,
    step: int,
    slice_schedule: Sequence[tuple[int, int]] | None,
    batch_size: int,
) -> tuple[int, list[int]]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("sds_batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("sds_batch_size must be positive.")

    if batch_size == 1:
        axis, index = select_slice(volume, step, slice_schedule)
        return axis, [index]

    if slice_schedule is not None:
        start = step * batch_size
        entries = slice_schedule[start : start + batch_size]
        if len(entries) != batch_size:
            raise ValueError(
                "slice_schedule must contain one entry per batched slice."
            )
        axes = [int(axis) for axis, _ in entries]
        if any(axis != axes[0] for axis in axes):
            raise ValueError("batched SDS slices must use the same axis.")
        axis = axes[0]
        _validate_axis(axis)
        indices = [int(index) for _, index in entries]
    else:
        axis = int(torch.randint(0, 3, (), device=volume.device).item())
        if batch_size > volume.shape[axis]:
            raise ValueError("sds_batch_size cannot exceed the selected axis length.")
        indices = torch.randperm(
            volume.shape[axis],
            device=volume.device,
        )[:batch_size].tolist()

    for index in indices:
        if index < 0 or index >= volume.shape[axis]:
            raise ValueError("slice_schedule index must be inside the selected axis.")
    if len(set(indices)) != len(indices):
        raise ValueError("batched SDS slices must be unique.")
    return axis, indices
