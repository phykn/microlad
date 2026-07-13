import torch


def volume_slices(
    volume: torch.Tensor,
    axis: int,
    *,
    num_phases: int,
) -> torch.Tensor:
    if volume.ndim != 5:
        raise ValueError("volume must have shape [B, C, D, H, W].")
    if volume.shape[1] != num_phases:
        raise ValueError("volume channel count must match num_phases.")
    depth, height, width = map(int, volume.shape[2:])
    if axis == 0:
        return volume.permute(0, 2, 1, 3, 4).reshape(-1, num_phases, height, width)
    if axis == 1:
        return volume.permute(0, 3, 1, 2, 4).reshape(-1, num_phases, depth, width)
    if axis == 2:
        return volume.permute(0, 4, 1, 2, 3).reshape(-1, num_phases, depth, height)
    raise ValueError("axis must be 0, 1, or 2.")
