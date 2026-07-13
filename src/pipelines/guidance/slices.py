import torch


def volume_slices(
    volume: torch.Tensor,
    axis: int,
    *,
    num_phases: int,
) -> torch.Tensor:
    """Flatten all planes of one axis into a 2D image batch."""

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


def critic_slices(
    volume: torch.Tensor,
    axis: int,
    *,
    num_phases: int,
    crop_size: int = 64,
    max_slices: int = 64,
) -> torch.Tensor:
    """Select bounded, deterministic 2D crops for the slice critic."""

    images = volume_slices(volume, axis, num_phases=num_phases)
    if images.shape[-2:] == (crop_size, crop_size):
        return images
    if min(images.shape[-2:]) < crop_size:
        raise ValueError("critic crop_size must fit inside every volume slice.")
    count = min(int(images.shape[0]), int(max_slices))
    indices = (
        torch.linspace(0, images.shape[0] - 1, steps=count, device=images.device)
        .round()
        .long()
    )
    selected = images.index_select(0, indices)
    max_row = int(selected.shape[-2]) - crop_size
    max_col = int(selected.shape[-1]) - crop_size
    positions = (
        (0, 0),
        (0, max_col),
        (max_row, 0),
        (max_row, max_col),
        (max_row // 2, max_col // 2),
    )
    return torch.stack(
        [
            image[:, row : row + crop_size, col : col + crop_size]
            for image, (row, col) in zip(
                selected,
                (positions[index % len(positions)] for index in range(count)),
                strict=True,
            )
        ]
    )


def transition_profile(probabilities: torch.Tensor, axis: int) -> torch.Tensor:
    """Expected categorical change rate between adjacent planes."""

    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape [C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    spatial_axis = axis + 1
    length = int(probabilities.shape[spatial_axis])
    if length < 2:
        raise ValueError("selected probability axis must contain at least two values.")
    before = probabilities.narrow(spatial_axis, 0, length - 1)
    after = probabilities.narrow(spatial_axis, 1, length - 1)
    same_probability = (before * after).sum(dim=0)
    reduce_axes = tuple(dimension for dimension in range(3) if dimension != axis)
    return (1.0 - same_probability).mean(dim=reduce_axes)
