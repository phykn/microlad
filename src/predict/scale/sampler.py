from collections.abc import Sequence

import torch

from src.predict.scale.denoise import denoise_tiled_plane


@torch.no_grad()
def sample_large_lmpdd(
    model: torch.nn.Module,
    ddpm,
    latent_shape: Sequence[int],
    *,
    tile_size: int,
    tile_overlap: int,
    device: str | torch.device,
) -> torch.Tensor:
    shape = _validate_latent_shape(latent_shape, tile_size=tile_size)
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    latent = torch.randn(shape, device=device)
    for pass_index, step in enumerate(range(int(ddpm.num_timesteps) - 1, -1, -1)):
        axis = pass_index % 3
        planes = _axis_to_planes(latent, axis)
        timesteps = torch.full(
            (planes.shape[0],),
            step,
            dtype=torch.long,
            device=planes.device,
        )
        planes = denoise_tiled_plane(
            model,
            ddpm,
            planes,
            timesteps,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
        latent = _planes_to_axis(planes, axis)

    return latent


def _validate_latent_shape(
    latent_shape: Sequence[int],
    *,
    tile_size: int,
) -> tuple[int, int, int, int]:
    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [C, D, H, W].")
    shape = tuple(int(value) for value in latent_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("latent_shape values must be positive.")
    if shape[1] != shape[2] or shape[1] != shape[3]:
        raise ValueError("large L-MPDD sampling requires a cubic latent shape.")
    if int(tile_size) <= 0:
        raise ValueError("tile_size must be positive.")
    if shape[1] < int(tile_size):
        raise ValueError("tile_size must fit inside latent spatial shape.")
    return shape


def _axis_to_planes(latent: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return latent.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return latent.permute(2, 0, 1, 3).contiguous()
    return latent.permute(3, 0, 1, 2).contiguous()


def _planes_to_axis(planes: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return planes.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return planes.permute(1, 2, 0, 3).contiguous()
    return planes.permute(1, 2, 3, 0).contiguous()
