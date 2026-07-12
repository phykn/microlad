import torch

from src.modeling.vae import get_downsample_factor
from src.modeling.phases.representation import probabilities_to_calibrated_labels
from src.pipelines.scaling.blending import blend_window
from src.pipelines.scaling.tiles import tile_grid
from src.common.tensors.validation import require_finite, require_float


@torch.no_grad()
def decode_large_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    _, depth, height, width = latent.shape
    factor = get_downsample_factor(vae)
    num_phases = getattr(vae, "num_phases", None)
    if (
        isinstance(num_phases, int)
        and not isinstance(num_phases, bool)
        and callable(getattr(vae, "decode_probs", None))
    ):
        return _decode_large_categorical(
            vae,
            latent,
            tile_overlap=tile_overlap,
            factor=factor,
            num_phases=num_phases,
        )

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

    return (volume / count.clamp_min(1)).float()


def _decode_large_categorical(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
    factor: int,
    num_phases: int,
) -> torch.Tensor:
    _, depth, height, width = latent.shape
    shape = (depth * factor, height * factor, width * factor)
    probabilities = torch.zeros(
        num_phases,
        *shape,
        dtype=torch.float32,
        device=latent.device,
    )
    count = torch.zeros(shape, dtype=torch.float32, device=latent.device)

    for index in range(depth):
        decoded = _decode_tiled_plane_probabilities(
            vae,
            latent[:, index, :, :],
            tile_overlap=tile_overlap,
            num_phases=num_phases,
        )
        start = index * factor
        probabilities[:, start : start + factor, :, :] += decoded.unsqueeze(1)
        count[start : start + factor, :, :] += 1

    for index in range(height):
        decoded = _decode_tiled_plane_probabilities(
            vae,
            latent[:, :, index, :],
            tile_overlap=tile_overlap,
            num_phases=num_phases,
        )
        start = index * factor
        probabilities[:, :, start : start + factor, :] += decoded.unsqueeze(2)
        count[:, start : start + factor, :] += 1

    for index in range(width):
        decoded = _decode_tiled_plane_probabilities(
            vae,
            latent[:, :, :, index],
            tile_overlap=tile_overlap,
            num_phases=num_phases,
        )
        start = index * factor
        probabilities[:, :, :, start : start + factor] += decoded.unsqueeze(3)
        count[:, :, start : start + factor] += 1

    probabilities = probabilities / count.clamp_min(1).unsqueeze(0)
    return probabilities_to_calibrated_labels(
        probabilities.unsqueeze(0),
        num_phases,
    )[0, 0].float()


def _validate_latent_volume(vae: torch.nn.Module, latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise ValueError("latent volume must have shape [C, D, H, W].")

    require_float("latent volume dtype", latent.dtype)
    require_finite("latent volume", latent)

    if latent.shape[0] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")

    latent_size = int(vae.latent_size)
    if any(size < latent_size for size in latent.shape[1:]):
        raise ValueError("latent spatial shape must be at least vae.latent_size.")

    get_downsample_factor(vae)


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
    factor = get_downsample_factor(vae)
    height, width = int(latent_plane.shape[1]), int(latent_plane.shape[2])
    out = torch.zeros(
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent_plane.device,
    )
    weight_sum = torch.zeros_like(out)
    if tile_overlap == 0:
        window = torch.ones(
            image_size,
            image_size,
            dtype=out.dtype,
            device=out.device,
        )
    else:
        window = blend_window(
            image_size,
            image_size,
            device=out.device,
            dtype=out.dtype,
        )

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

        require_finite("decoded", decoded)

        out_row = row * factor
        out_col = col * factor
        out[
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += decoded[0, 0].float() * window
        weight_sum[
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += window

    return out / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)


def _decode_tiled_plane_probabilities(
    vae: torch.nn.Module,
    latent_plane: torch.Tensor,
    *,
    tile_overlap: int,
    num_phases: int,
) -> torch.Tensor:
    if latent_plane.ndim != 3:
        raise ValueError("latent_plane must have shape [C, H, W].")

    tile_size = int(vae.latent_size)
    image_size = int(vae.image_size)
    factor = get_downsample_factor(vae)
    height, width = int(latent_plane.shape[1]), int(latent_plane.shape[2])
    out = torch.zeros(
        num_phases,
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent_plane.device,
    )
    weight_sum = torch.zeros_like(out[0])
    window = (
        torch.ones(image_size, image_size, dtype=out.dtype, device=out.device)
        if tile_overlap == 0
        else blend_window(
            image_size,
            image_size,
            device=out.device,
            dtype=out.dtype,
        )
    )

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
        decoded = vae.decode_probs(latent_tile)
        expected_shape = (1, num_phases, image_size, image_size)
        if decoded.shape != expected_shape:
            raise ValueError(
                "decode_probs output must have shape [1, num_phases, H, W]."
            )
        require_finite("decoded probabilities", decoded)

        out_row = row * factor
        out_col = col * factor
        out[
            :,
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += decoded[0].float() * window.unsqueeze(0)
        weight_sum[
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += window

    return out / weight_sum.clamp_min(
        torch.finfo(weight_sum.dtype).tiny
    ).unsqueeze(0)
