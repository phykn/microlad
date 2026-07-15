import torch

from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.modeling.vae import get_downsample_factor
from src.pipeline.predict.guidance.conditioning.model import AnchorSlice
from src.pipeline.predict.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def decode_large_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    tile_overlap: int,
    batch_size: int | None = 16,
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
    batch_size: int | None = 16,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    if batch_size is not None:
        require_int("batch_size", batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")
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


def decode_anchor_patch(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    anchor: AnchorSlice,
    *,
    target_size: int,
    num_phases: int,
    tile_overlap: int,
    batch_size: int | None = None,
    crop_start: tuple[int, int] = (0, 0),
    crop_size: int | None = None,
) -> torch.Tensor:
    factor = get_downsample_factor(vae)
    output_size = int(latent.shape[1]) * factor
    if target_size == int(vae.image_size):
        start = (output_size - target_size) // 2
        if start % factor != 0:
            raise ValueError("scale anchor patch must align to the latent grid.")
    elif target_size == output_size:
        start = 0
    else:
        raise ValueError("scale anchor image must match vae.image_size or output size.")

    return _decode_probability_patch(
        vae,
        latent,
        axis=int(anchor.axis),
        index=int(anchor.index),
        start=start,
        target_size=target_size,
        num_phases=num_phases,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        crop_start=crop_start,
        crop_size=crop_size,
    )


def decode_consensus_patch(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    axis: int,
    index: int,
    num_phases: int,
    tile_overlap: int,
    batch_size: int | None = None,
    crop_start: tuple[int, int] = (0, 0),
    crop_size: int | None = None,
) -> torch.Tensor:
    """Decode one patch exactly as the final tri-axis consensus volume."""
    _validate_latent_volume(vae, latent)
    output_size = int(latent.shape[1]) * get_downsample_factor(vae)
    return _decode_probability_patch(
        vae,
        latent,
        axis=axis,
        index=index,
        start=0,
        target_size=output_size,
        num_phases=num_phases,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        crop_start=crop_start,
        crop_size=crop_size,
    )


def _decode_probability_patch(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    axis: int,
    index: int,
    start: int,
    target_size: int,
    num_phases: int,
    tile_overlap: int,
    batch_size: int | None,
    crop_start: tuple[int, int] = (0, 0),
    crop_size: int | None = None,
) -> torch.Tensor:
    latent_size = int(latent.shape[1])
    output_size = latent_size * get_downsample_factor(vae)
    if crop_size is None:
        crop_size = target_size
    crop_row, crop_col = crop_start
    if (
        axis not in (0, 1, 2)
        or index < 0
        or index >= output_size
        or start < 0
        or start + target_size > output_size
        or crop_size <= 0
        or crop_row < 0
        or crop_col < 0
        or crop_row + crop_size > target_size
        or crop_col + crop_size > target_size
    ):
        raise ValueError("probability patch lies outside the output volume.")

    patch_dims = [dimension for dimension in range(3) if dimension != axis]
    row, col = torch.meshgrid(
        torch.arange(crop_size, device=latent.device),
        torch.arange(crop_size, device=latent.device),
        indexing="ij",
    )
    local = {patch_dims[0]: row, patch_dims[1]: col}
    global_coords = {
        axis: torch.full_like(row, index),
        patch_dims[0]: row + start + crop_row,
        patch_dims[1]: col + start + crop_col,
    }
    contributions = []
    for decode_axis in range(3):
        axis_coords = (
            torch.tensor([index], device=latent.device)
            if decode_axis == axis
            else torch.arange(
                start + crop_start[patch_dims.index(decode_axis)],
                start + crop_start[patch_dims.index(decode_axis)] + crop_size,
                device=latent.device,
            )
        )
        positions = (
            (axis_coords.to(latent.dtype) + 0.5) * latent_size / output_size - 0.5
        ).clamp(0.0, float(latent_size - 1))
        lower = positions.floor().long()
        upper = (lower + 1).clamp_max(latent_size - 1)
        plane_indices = torch.unique(torch.cat([lower, upper]), sorted=True)
        planes = torch.stack(
            [
                _latent_plane(latent, decode_axis, int(plane_index))
                for plane_index in plane_indices.tolist()
            ]
        )
        decoded = decode_tiled_planes(
            vae,
            planes,
            tile_overlap=tile_overlap,
            num_phases=num_phases,
            batch_size=batch_size,
        )

        lower_slot = torch.searchsorted(plane_indices, lower)
        upper_slot = torch.searchsorted(plane_indices, upper)
        weight = (positions - lower).view(-1, 1, 1, 1)
        interpolated = (
            decoded[lower_slot] * (1.0 - weight)
            + decoded[upper_slot] * weight
        )
        axis_slot = torch.zeros_like(row) if decode_axis == axis else local[decode_axis]
        spatial_dims = [dimension for dimension in range(3) if dimension != decode_axis]
        values = interpolated.permute(0, 2, 3, 1)[
            axis_slot,
            global_coords[spatial_dims[0]],
            global_coords[spatial_dims[1]],
        ]
        contributions.append(values.movedim(-1, 0))

    tiny = torch.finfo(contributions[0].dtype).tiny
    logits = torch.stack(contributions).clamp_min(tiny).log().mean(dim=0)
    return logits.softmax(dim=0)


def _latent_plane(latent: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    if axis == 0:
        return latent[:, index, :, :]
    if axis == 1:
        return latent[:, :, index, :]
    if axis == 2:
        return latent[:, :, :, index]
    raise ValueError("axis must be 0, 1, or 2.")


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
    batch_size: int | None,
) -> torch.Tensor:
    plane_count = int(latent.shape[axis + 1])
    chunk_size = plane_count if batch_size is None else batch_size
    decoded = None
    for start in range(0, plane_count, chunk_size):
        stop = min(start + chunk_size, plane_count)
        batch = _latent_plane_batch(latent, axis, start, stop)
        probabilities = decode_tiled_planes(
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


def decode_tiled_planes(
    vae: torch.nn.Module,
    latent_planes: torch.Tensor,
    *,
    tile_overlap: int,
    num_phases: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    if latent_planes.ndim != 4:
        raise ValueError("latent_planes must have shape [B, C, H, W].")
    if batch_size is not None:
        require_int("batch_size", batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or None.")

    tile_size = int(vae.latent_size)
    image_size = int(vae.image_size)
    factor = get_downsample_factor(vae)
    plane_count, _, height, width = latent_planes.shape
    out = torch.zeros(
        plane_count,
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
        chunk_size = latent_tile.shape[0] if batch_size is None else batch_size
        decoded = torch.cat(
            [
                vae.decode_probs(latent_tile[start : start + chunk_size])
                for start in range(0, latent_tile.shape[0], chunk_size)
            ]
        )
        expected = (plane_count, num_phases, image_size, image_size)
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
    batch_size: int | None,
) -> None:
    input_size = int(planes.shape[0])
    tiny = torch.finfo(planes.dtype).tiny
    chunk_size = output_size if batch_size is None else batch_size
    for start in range(0, output_size, chunk_size):
        stop = min(start + chunk_size, output_size)
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
