import torch

from src.pipelines.scaling.blending import blend_window
from src.pipelines.scaling.tiles import tile_grid
from src.common.tensors.validation import require_finite


@torch.no_grad()
def reconstruct_anchor_target(
    vae: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_overlap: int = 0,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("anchor target image must have shape [1, 1, H, W].")

    require_finite("anchor target image", image)

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

    for row, col in tile_grid(
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
    require_finite("encoded anchor target", mu)

    recon = vae.decode(mu)
    if recon.shape != image.shape:
        raise ValueError("reconstructed anchor target must match input shape.")

    require_finite("reconstructed anchor target", recon)

    return recon.detach()
