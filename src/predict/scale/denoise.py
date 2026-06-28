import torch

from src.predict.scale.tiles import tile_grid


@torch.no_grad()
def denoise_tiled_plane(
    model: torch.nn.Module,
    ddpm,
    planes: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    if planes.ndim != 4:
        raise ValueError("planes must have shape [B, C, H, W].")
    if timesteps.ndim != 1 or timesteps.shape[0] != planes.shape[0]:
        raise ValueError("timesteps must have shape [B].")

    _, _, height, width = planes.shape
    out = torch.zeros_like(planes)
    count = torch.zeros(
        (1, 1, height, width),
        dtype=planes.dtype,
        device=planes.device,
    )

    for row, col in tile_grid(height, width, tile_size=tile_size, overlap=overlap):
        patch = planes[:, :, row : row + tile_size, col : col + tile_size]
        denoised = ddpm.p_sample(model, patch, timesteps)
        if denoised.shape != patch.shape:
            raise ValueError("ddpm.p_sample output must match input patch shape.")
        out[:, :, row : row + tile_size, col : col + tile_size] += denoised
        count[:, :, row : row + tile_size, col : col + tile_size] += 1

    return out / count.clamp_min(1)
