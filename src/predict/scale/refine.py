import torch

from src.predict.scale.tiles import tile_grid
from src.predict.validation import validate_finite_tensor, validate_floating_dtype


@torch.no_grad()
def refine_large_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    tile_overlap: int = 0,
) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    _validate_volume(volume)

    refined = volume.clamp(-1.0, 1.0).float()
    if steps == 0:
        return refined

    vae.eval()
    for _ in range(steps):
        refined = _refine_once(refined, vae, tile_overlap=tile_overlap)

    return refined


def _refine_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    depth, height, width = volume.shape
    out = torch.zeros_like(volume)
    count = torch.zeros_like(volume)

    for index in range(depth):
        refined = _refine_tiled_plane(
            volume[index, :, :],
            vae,
            tile_overlap=tile_overlap,
        )
        out[index, :, :] += refined
        count[index, :, :] += 1

    for index in range(height):
        refined = _refine_tiled_plane(
            volume[:, index, :],
            vae,
            tile_overlap=tile_overlap,
        )
        out[:, index, :] += refined
        count[:, index, :] += 1

    for index in range(width):
        refined = _refine_tiled_plane(
            volume[:, :, index],
            vae,
            tile_overlap=tile_overlap,
        )
        out[:, :, index] += refined
        count[:, :, index] += 1

    return (out / count.clamp_min(1)).clamp(-1.0, 1.0).float()


def _refine_tiled_plane(
    image: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")

    tile_size = int(vae.image_size)
    height, width = int(image.shape[0]), int(image.shape[1])
    out = torch.zeros_like(image, dtype=torch.float32)
    count = torch.zeros_like(image, dtype=torch.float32)

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        tile = image[row : row + tile_size, col : col + tile_size].view(
            1,
            1,
            tile_size,
            tile_size,
        )
        mu, _ = vae.encode(tile)

        if mu.ndim != 4:
            raise ValueError("encode output must have shape [B, C, H, W].")

        validate_finite_tensor("encoded latent", mu)

        decoded = vae.decode(mu)
        if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
            raise ValueError("decode output must have shape [1, 1, H, W].")

        if decoded.shape[-2:] != (tile_size, tile_size):
            raise ValueError("decode output spatial shape must match vae.image_size.")

        validate_finite_tensor("decoded tile", decoded)

        out[row : row + tile_size, col : col + tile_size] += decoded[0, 0].float()
        count[row : row + tile_size, col : col + tile_size] += 1

    return out / count.clamp_min(1)


def _validate_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    validate_floating_dtype("volume dtype", volume.dtype)
    validate_finite_tensor("volume", volume)

    depth, height, width = volume.shape
    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("large volume refinement requires a cubic volume.")
