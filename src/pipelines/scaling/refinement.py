import torch

from src.pipelines.scaling.blending import blend_window
from src.pipelines.scaling.tiles import tile_grid
from src.common.tensors.validation import require_finite, require_float


@torch.no_grad()
def refine_large_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    tile_overlap: int = 0,
    tile_batch_size: int = 16,
) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    _validate_batch_size(tile_batch_size)
    _validate_volume(volume)

    refined = volume.float()
    if steps == 0:
        return refined

    vae.eval()
    for _ in range(steps):
        refined = _refine_once(
            refined,
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )

    return refined


def _refine_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int,
) -> torch.Tensor:
    depth, height, width = volume.shape
    out = torch.zeros_like(volume)
    count = torch.zeros_like(volume)

    for index in range(depth):
        refined = _refine_tiled_plane(
            volume[index, :, :],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )
        out[index, :, :] += refined
        count[index, :, :] += 1

    for index in range(height):
        refined = _refine_tiled_plane(
            volume[:, index, :],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )
        out[:, index, :] += refined
        count[:, index, :] += 1

    for index in range(width):
        refined = _refine_tiled_plane(
            volume[:, :, index],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )
        out[:, :, index] += refined
        count[:, :, index] += 1

    return (out / count.clamp_min(1)).float()


def _refine_tiled_plane(
    image: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int,
) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")

    _validate_batch_size(tile_batch_size)

    tile_size = int(vae.image_size)
    height, width = int(image.shape[0]), int(image.shape[1])
    out = torch.zeros_like(image, dtype=torch.float32)
    weight_sum = torch.zeros_like(image, dtype=torch.float32)
    if tile_overlap == 0:
        window = torch.ones(
            tile_size,
            tile_size,
            dtype=out.dtype,
            device=out.device,
        )
    else:
        window = blend_window(
            tile_size,
            tile_size,
            device=out.device,
            dtype=out.dtype,
        )

    positions = list(
        tile_grid(
            height,
            width,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
    )
    for start in range(0, len(positions), tile_batch_size):
        chunk = positions[start : start + tile_batch_size]
        batch = torch.stack(
            [
                image[row : row + tile_size, col : col + tile_size]
                for row, col in chunk
            ],
            dim=0,
        ).view(len(chunk), 1, tile_size, tile_size)
        decoded = _encode_decode_tiles(vae, batch, tile_size)

        for tile, (row, col) in zip(decoded[:, 0], chunk):
            out[row : row + tile_size, col : col + tile_size] += tile * window
            weight_sum[row : row + tile_size, col : col + tile_size] += window

    return out / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)


def _encode_decode_tiles(
    vae: torch.nn.Module,
    tiles: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    mu, _ = vae.encode(tiles)

    if mu.ndim != 4:
        raise ValueError("encode output must have shape [B, C, H, W].")

    if mu.shape[0] != tiles.shape[0]:
        raise ValueError("encode output batch size must match input tiles.")

    require_finite("encoded latent", mu)

    decoded = vae.decode(mu)
    if decoded.ndim != 4 or decoded.shape[:2] != (tiles.shape[0], 1):
        raise ValueError("decode output must have shape [B, 1, H, W].")

    if decoded.shape[-2:] != (tile_size, tile_size):
        raise ValueError("decode output spatial shape must match vae.image_size.")

    require_finite("decoded tile", decoded)

    return decoded.float()


def _validate_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    require_float("volume dtype", volume.dtype)
    require_finite("volume", volume)

    depth, height, width = volume.shape
    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("large volume refinement requires a cubic volume.")


def _validate_batch_size(tile_batch_size: int) -> None:
    if not isinstance(tile_batch_size, int) or isinstance(tile_batch_size, bool):
        raise ValueError("tile_batch_size must be an integer.")

    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive.")
