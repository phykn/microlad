import torch


def sample_slices(
    volume: torch.Tensor,
    *,
    count: int,
    crop_size: int,
) -> torch.Tensor:
    """Samples balanced XY, XZ, and YZ latent crops."""
    if volume.ndim != 5:
        raise ValueError("latent volume must have shape [B, C, D, H, W].")
    if count <= 0 or crop_size <= 0:
        raise ValueError("count and crop_size must be positive.")

    slices = []
    for sample in range(count):
        batch = sample % int(volume.shape[0])
        axis = sample % 3
        index = int(
            torch.randint(0, volume.shape[axis + 2], (), device=volume.device).item()
        )
        plane = volume[batch].select(axis + 1, index)
        if min(plane.shape[-2:]) < crop_size:
            raise ValueError("latent slices are smaller than crop_size.")
        max_row = int(plane.shape[-2]) - crop_size
        max_col = int(plane.shape[-1]) - crop_size
        row = int(torch.randint(0, max_row + 1, (), device=volume.device).item())
        col = int(torch.randint(0, max_col + 1, (), device=volume.device).item())
        slices.append(plane[:, row : row + crop_size, col : col + crop_size])
    return torch.stack(slices)
