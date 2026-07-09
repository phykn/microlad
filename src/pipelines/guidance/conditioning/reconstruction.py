import torch

from src.pipelines.scaling.blending import blend_window
from src.common.tensors.validation import validate_finite_tensor


@torch.no_grad()
def reconstruct_anchor_target(
    vae: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_overlap: int = 0,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("anchor target image must have shape [1, 1, H, W].")

    validate_finite_tensor("anchor target image", image)

    image_size = int(vae.image_size)
    height, width = int(image.shape[-2]), int(image.shape[-1])

    if (height, width) == (image_size, image_size):
        return _reconstruct_patch(vae, image)

    out = torch.zeros_like(image)
    weight_sum = torch.zeros_like(image)
    if tile_overlap == 0:
        window = torch.ones(
            1,
            1,
            image_size,
            image_size,
            dtype=image.dtype,
            device=image.device,
        )
    else:
        window = blend_window(
            image_size,
            image_size,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, image_size, image_size)

    for row, col in _tile_grid(
        height,
        width,
        tile_size=image_size,
        overlap=tile_overlap,
    ):
        patch = image[:, :, row : row + image_size, col : col + image_size]
        recon = _reconstruct_patch(vae, patch)
        out[:, :, row : row + image_size, col : col + image_size] += recon * window
        weight_sum[:, :, row : row + image_size, col : col + image_size] += window

    return (
        out / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)
    ).detach()


def _reconstruct_patch(vae: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    vae.eval()
    mu, _ = vae.encode(image)
    validate_finite_tensor("encoded anchor target", mu)

    recon = vae.decode(mu)
    if recon.shape != image.shape:
        raise ValueError("reconstructed anchor target must match input shape.")

    validate_finite_tensor("reconstructed anchor target", recon)

    return recon.detach()


def _tile_grid(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
):
    for row in _tile_starts(height, tile_size=tile_size, overlap=overlap):
        for col in _tile_starts(width, tile_size=tile_size, overlap=overlap):
            yield row, col


def _tile_starts(size: int, *, tile_size: int, overlap: int) -> list[int]:
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size.")

    if tile_size > size:
        raise ValueError("tile_size must fit inside size.")

    stride = tile_size - overlap
    starts = list(range(0, size - tile_size + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)

    return starts
