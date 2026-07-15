import torch


def slice_volume(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return volume.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return volume.permute(2, 0, 1, 3).contiguous()
    if axis == 2:
        return volume.permute(3, 0, 1, 2).contiguous()
    raise ValueError("axis must be 0, 1, or 2.")


def merge_planes(planes: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return planes.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return planes.permute(1, 2, 0, 3).contiguous()
    if axis == 2:
        return planes.permute(1, 2, 3, 0).contiguous()
    raise ValueError("axis must be 0, 1, or 2.")
