import torch

from src.predict.scale.tiles import tile_grid


@torch.no_grad()
def decode_large_latent_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    _, depth, height, width = latent.shape
    factor = _downsample_factor(vae)
    volume = torch.zeros(
        depth * factor,
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent.device,
    )
    count = torch.zeros_like(volume)

    for index in range(depth):
        decoded = _decode_tiled_plane(
            vae,
            latent[:, index, :, :],
            tile_overlap=tile_overlap,
        )
        start = index * factor
        volume[start : start + factor, :, :] += decoded.unsqueeze(0)
        count[start : start + factor, :, :] += 1

    for index in range(height):
        decoded = _decode_tiled_plane(
            vae,
            latent[:, :, index, :],
            tile_overlap=tile_overlap,
        )
        start = index * factor
        volume[:, start : start + factor, :] += decoded.unsqueeze(1)
        count[:, start : start + factor, :] += 1

    for index in range(width):
        decoded = _decode_tiled_plane(
            vae,
            latent[:, :, :, index],
            tile_overlap=tile_overlap,
        )
        start = index * factor
        volume[:, :, start : start + factor] += decoded.unsqueeze(2)
        count[:, :, start : start + factor] += 1

    return (volume / count.clamp_min(1)).clamp(-1.0, 1.0).float()


def _validate_latent_volume(vae: torch.nn.Module, latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise ValueError("latent volume must have shape [C, D, H, W].")
    if latent.shape[0] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")
    latent_size = int(vae.latent_size)
    if any(size < latent_size for size in latent.shape[1:]):
        raise ValueError("latent spatial shape must be at least vae.latent_size.")


def _decode_tiled_plane(
    vae: torch.nn.Module,
    latent_plane: torch.Tensor,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    if latent_plane.ndim != 3:
        raise ValueError("latent_plane must have shape [C, H, W].")

    tile_size = int(vae.latent_size)
    image_size = int(vae.image_size)
    factor = _downsample_factor(vae)
    height, width = int(latent_plane.shape[1]), int(latent_plane.shape[2])
    out = torch.zeros(
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent_plane.device,
    )
    count = torch.zeros_like(out)

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        latent_tile = latent_plane[
            :,
            row : row + tile_size,
            col : col + tile_size,
        ].unsqueeze(0)
        decoded = vae.decode(latent_tile)
        if decoded.ndim != 4 or decoded.shape[0] != 1 or decoded.shape[1] != 1:
            raise ValueError("decode output must have shape [1, 1, H, W].")
        if decoded.shape[-2:] != (image_size, image_size):
            raise ValueError("decode output spatial shape must match vae.image_size.")

        out_row = row * factor
        out_col = col * factor
        out[
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += decoded[0, 0].float()
        count[
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += 1

    return out / count.clamp_min(1)


def _downsample_factor(vae: torch.nn.Module) -> int:
    return int(
        getattr(
            vae,
            "downsample_factor",
            int(vae.image_size) // int(vae.latent_size),
        )
    )
