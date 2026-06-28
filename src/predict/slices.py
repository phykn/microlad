from collections.abc import Sequence

import torch


def extract_slice(volume: torch.Tensor, axis: int, index: int) -> torch.Tensor:
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
    if axis == 0:
        volume[index, :, :] = image
    elif axis == 1:
        volume[:, index, :] = image
    else:
        volume[:, :, index] = image


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
