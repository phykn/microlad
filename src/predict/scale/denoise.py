import torch

from src.predict.scale.tiles import tile_grid
from src.predict.validation import validate_finite_tensor, validate_floating_dtype


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

    validate_floating_dtype("planes dtype", planes.dtype)
    validate_finite_tensor("planes", planes)

    if timesteps.ndim != 1 or timesteps.shape[0] != planes.shape[0]:
        raise ValueError("timesteps must have shape [B].")

    if timesteps.dtype != torch.long:
        raise ValueError("timesteps must be integer tensors.")

    _, _, height, width = planes.shape
    mean_out = torch.zeros_like(planes)
    count = torch.zeros(
        (1, 1, height, width),
        dtype=planes.dtype,
        device=planes.device,
    )

    for row, col in tile_grid(height, width, tile_size=tile_size, overlap=overlap):
        patch = planes[:, :, row : row + tile_size, col : col + tile_size]
        mean_tile = ddpm.p_mean(model, patch, timesteps)

        if mean_tile.shape != patch.shape:
            raise ValueError("ddpm.p_mean output must match input patch shape.")

        validate_finite_tensor("p_mean output", mean_tile)

        mean_out[:, :, row : row + tile_size, col : col + tile_size] += mean_tile
        count[:, :, row : row + tile_size, col : col + tile_size] += 1

    mean_out = mean_out / count.clamp_min(1)
    noise = torch.randn_like(mean_out)
    shape = (timesteps.shape[0],) + (1,) * (mean_out.ndim - 1)
    noise = torch.where(timesteps.view(shape) > 0, noise, torch.zeros_like(noise))
    variance = ddpm._expand(ddpm.posterior_variance, timesteps, mean_out.ndim)
    denoised = mean_out + torch.sqrt(variance) * noise

    validate_finite_tensor("denoised", denoised)

    return denoised
