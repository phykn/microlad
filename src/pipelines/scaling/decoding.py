import torch

from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.modeling.vae import get_downsample_factor
from src.pipelines.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def decode_large_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
    batch_size: int = 16,
) -> torch.Tensor:
    probabilities = decode_large_volume_probabilities(
        vae,
        latent,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
    )
    return probabilities_to_calibrated_labels(
        probabilities,
        int(vae.num_phases),
    )[0, 0].float()


@torch.no_grad()
def decode_large_volume_probabilities(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
    num_phases: int | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    require_int("batch_size", batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    vae.eval()

    if num_phases is None:
        num_phases = getattr(vae, "num_phases", None)
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if num_phases != getattr(vae, "num_phases", None):
        raise ValueError("num_phases must match vae.num_phases.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise ValueError("categorical volume decoding requires vae.decode_probs.")

    factor = get_downsample_factor(vae)
    depth, height, width = map(int, latent.shape[1:])
    log_sum = torch.zeros(
        num_phases,
        depth * factor,
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent.device,
    )

    for axis in range(3):
        planes = _decode_axis_planes(
            vae,
            latent,
            axis=axis,
            tile_overlap=tile_overlap,
            num_phases=num_phases,
            batch_size=batch_size,
        )
        _accumulate_axis(
            log_sum,
            planes,
            axis=axis,
            output_size=latent.shape[axis + 1] * factor,
            batch_size=batch_size,
        )

    log_sum.div_(3.0)
    log_sum.sub_(log_sum.amax(dim=0, keepdim=True))
    log_sum.exp_()
    log_sum.div_(log_sum.sum(dim=0, keepdim=True))
    require_finite("decoded volume probabilities", log_sum)
    return log_sum.unsqueeze(0)


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


def _decode_axis_planes(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    axis: int,
    tile_overlap: int,
    num_phases: int,
    batch_size: int,
) -> torch.Tensor:
    plane_count = int(latent.shape[axis + 1])
    decoded = None
    for start in range(0, plane_count, batch_size):
        stop = min(start + batch_size, plane_count)
        batch = _latent_plane_batch(latent, axis, start, stop)
        probabilities = _decode_tiled_plane_probabilities(
            vae,
            batch,
            tile_overlap=tile_overlap,
            num_phases=num_phases,
        )
        if decoded is None:
            decoded = torch.empty(
                plane_count,
                *probabilities.shape[1:],
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
        decoded[start:stop] = probabilities

    if decoded is None:
        raise RuntimeError("large-volume decoding produced no planes.")
    return decoded


def _latent_plane_batch(
    latent: torch.Tensor,
    axis: int,
    start: int,
    stop: int,
) -> torch.Tensor:
    if axis == 0:
        return latent[:, start:stop, :, :].permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return latent[:, :, start:stop, :].permute(2, 0, 1, 3).contiguous()
    if axis == 2:
        return latent[:, :, :, start:stop].permute(3, 0, 1, 2).contiguous()
    raise ValueError("axis must be 0, 1, or 2.")


def _decode_tiled_plane_probabilities(
    vae: torch.nn.Module,
    latent_planes: torch.Tensor,
    *,
    tile_overlap: int,
    num_phases: int,
) -> torch.Tensor:
    if latent_planes.ndim != 4:
        raise ValueError("latent_planes must have shape [B, C, H, W].")

    tile_size = int(vae.latent_size)
    image_size = int(vae.image_size)
    factor = get_downsample_factor(vae)
    batch_size, _, height, width = latent_planes.shape
    out = torch.zeros(
        batch_size,
        num_phases,
        height * factor,
        width * factor,
        dtype=torch.float32,
        device=latent_planes.device,
    )
    weight_sum = torch.zeros_like(out[:, :1])
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
        latent_tile = latent_planes[
            :,
            :,
            row : row + tile_size,
            col : col + tile_size,
        ]
        decoded = vae.decode_probs(latent_tile)
        expected = (batch_size, num_phases, image_size, image_size)
        if decoded.shape != expected:
            raise ValueError(
                "decode_probs output must have shape [B, num_phases, H, W]."
            )
        require_finite("decoded probabilities", decoded)

        out_row = row * factor
        out_col = col * factor
        out[
            :,
            :,
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += decoded.float() * window.view(1, 1, image_size, image_size)
        weight_sum[
            :,
            :,
            out_row : out_row + image_size,
            out_col : out_col + image_size,
        ] += window.view(1, 1, image_size, image_size)

    return out / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)


def _accumulate_axis(
    log_sum: torch.Tensor,
    planes: torch.Tensor,
    *,
    axis: int,
    output_size: int,
    batch_size: int,
) -> None:
    input_size = int(planes.shape[0])
    tiny = torch.finfo(planes.dtype).tiny
    for start in range(0, output_size, batch_size):
        stop = min(start + batch_size, output_size)
        positions = (
            (torch.arange(start, stop, device=planes.device, dtype=planes.dtype) + 0.5)
            * input_size
            / output_size
            - 0.5
        ).clamp(0.0, float(input_size - 1))
        lower = positions.floor().long()
        upper = (lower + 1).clamp_max(input_size - 1)
        weight = (positions - lower).view(-1, 1, 1, 1)
        interpolated = planes[lower] * (1.0 - weight) + planes[upper] * weight
        values = interpolated.permute(1, 0, 2, 3).clamp_min(tiny).log()

        if axis == 0:
            log_sum[:, start:stop, :, :] += values
        elif axis == 1:
            log_sum[:, :, start:stop, :] += values.permute(0, 2, 1, 3)
        elif axis == 2:
            log_sum[:, :, :, start:stop] += values.permute(0, 2, 3, 1)
        else:
            raise ValueError("axis must be 0, 1, or 2.")
