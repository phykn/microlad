from collections.abc import Sequence

import numpy as np
import torch

from src.modeling.vae import get_downsample_factor
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.reconstruction import reconstruct_target
from src.pipelines.guidance.conditioning.validation import validate_anchor, validate_anchors
from src.pipelines.scaling.tiles import tile_grid
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.common.tensors.validation import require_finite


def center_start(*, volume_size: int, base_size: int) -> int:
    volume_size = int(volume_size)
    base_size = int(base_size)

    if volume_size < base_size:
        raise ValueError("volume_size must be at least base_size.")

    return (volume_size - base_size) // 2


def _aligned_center_start(
    *,
    volume_size: int,
    base_size: int,
    downsample_factor: int,
) -> int:
    start = center_start(volume_size=volume_size, base_size=base_size)
    factor = int(downsample_factor)

    if factor <= 0:
        raise ValueError("downsample_factor must be positive.")

    if start % factor != 0:
        raise ValueError("anchor center must align to the VAE latent grid.")

    return start


def shift_anchor_slices(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    base_size: int,
    downsample_factor: int | None = None,
) -> list[tuple[int, int]]:
    if downsample_factor is None:
        start = center_start(volume_size=volume_size, base_size=base_size)
    else:
        start = _aligned_center_start(
            volume_size=volume_size,
            base_size=base_size,
            downsample_factor=downsample_factor,
        )

    validate_anchors(anchors or [], (base_size, base_size, base_size))

    return [(int(anchor.axis), start + int(anchor.index)) for anchor in anchors or []]


def encode_scale_anchors(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
    tile_overlap: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not anchors:
        return None, None

    factor = get_downsample_factor(vae)
    volume_size = int(volume_size)
    if volume_size % factor != 0:
        raise ValueError("volume_size must be divisible by VAE downsample factor.")

    image_size = int(vae.image_size)
    anchor_scopes = [
        _scale_anchor_scope(
            anchor,
            base_size=image_size,
            volume_size=volume_size,
        )
        for anchor in anchors
    ]
    needs_base_center = any(scope == "base" for scope in anchor_scopes)

    image_start = (
        _aligned_center_start(
            volume_size=volume_size,
            base_size=image_size,
            downsample_factor=factor,
        )
        if needs_base_center
        else 0
    )
    latent_size = volume_size // factor
    start = image_start // factor

    latent = torch.zeros(
        (int(vae.latent_ch), latent_size, latent_size, latent_size),
        device=device,
    )
    mask = torch.zeros_like(latent)
    written_planes: set[tuple[int, int, int, int, int]] = set()

    for anchor, scope in zip(anchors, anchor_scopes, strict=True):
        if scope == "base":
            encoded = _encode_anchor(
                vae,
                anchor,
                num_phases=num_phases,
                segment=segment,
                device=device,
            )
            index = start + int(anchor.index) // factor
            plane_start = start
        elif scope == "volume":
            encoded = _encode_large_anchor(
                vae,
                anchor,
                volume_size=volume_size,
                num_phases=num_phases,
                segment=segment,
                device=device,
                tile_overlap=tile_overlap,
            )
            index = int(anchor.index) // factor
            plane_start = 0
        else:
            raise ValueError("anchor image size must match vae.image_size or volume_size.")

        key = (
            int(anchor.axis),
            int(index),
            int(plane_start),
            int(encoded.shape[1]),
            int(encoded.shape[2]),
        )
        if key in written_planes:
            raise ValueError("anchor slices collapse to the same latent plane.")

        written_planes.add(key)

        _write_anchor_plane(
            latent,
            mask,
            encoded,
            axis=int(anchor.axis),
            index=index,
            start=plane_start,
        )

    return latent, mask


def build_scale_targets(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    base_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
    dtype: torch.dtype,
    downsample_factor: int | None = None,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    if not anchors:
        return {}, {}

    volume_size = int(volume_size)
    base_size = int(base_size)
    validate_anchors(anchors, (base_size, base_size, base_size))

    if downsample_factor is None:
        start = center_start(volume_size=volume_size, base_size=base_size)
    else:
        start = _aligned_center_start(
            volume_size=volume_size,
            base_size=base_size,
            downsample_factor=downsample_factor,
        )

    targets: dict[tuple[int, int], torch.Tensor] = {}
    masks: dict[tuple[int, int], torch.Tensor] = {}

    for anchor in anchors:
        image = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        ).to(device=device, dtype=dtype)
        image = reconstruct_target(vae, image)[0, 0]

        target = torch.zeros((volume_size, volume_size), device=device, dtype=dtype)
        mask = torch.zeros_like(target)
        target[start : start + base_size, start : start + base_size] = image
        mask[start : start + base_size, start : start + base_size] = 1

        targets[(int(anchor.axis), start + int(anchor.index))] = target
        masks[(int(anchor.axis), start + int(anchor.index))] = mask

    return targets, masks


def _scale_anchor_scope(
    anchor: AnchorSlice,
    *,
    base_size: int,
    volume_size: int,
) -> str:
    if isinstance(anchor.image, np.ndarray) and anchor.image.ndim == 2:
        image_shape = tuple(int(size) for size in anchor.image.shape)

        if image_shape == (base_size, base_size):
            validate_anchor(anchor, (base_size, base_size, base_size))
            return "base"

        if image_shape == (volume_size, volume_size):
            validate_anchor(anchor, (volume_size, volume_size, volume_size))
            return "volume"

        raise ValueError("anchor image size must match vae.image_size or volume_size.")

    validate_anchor(anchor, (base_size, base_size, base_size))
    raise ValueError("anchor image size must match vae.image_size or volume_size.")


def _encode_anchor(
    vae: torch.nn.Module,
    anchor: AnchorSlice,
    *,
    num_phases: int,
    segment: bool,
    device: torch.device,
) -> torch.Tensor:
    image = prepare_anchor_image(
        anchor.image,
        num_phases=num_phases,
        segment=segment,
    ).to(device=device)

    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(image)

    expected = torch.Size(
        [1, int(vae.latent_ch), int(vae.latent_size), int(vae.latent_size)]
    )
    if mu.shape != expected:
        raise ValueError(f"encoded anchor latent must have shape {tuple(expected)}.")

    require_finite("encoded anchor latent", mu)

    return mu[0].detach()


def _encode_large_anchor(
    vae: torch.nn.Module,
    anchor: AnchorSlice,
    *,
    volume_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
    tile_overlap: int,
) -> torch.Tensor:
    factor = get_downsample_factor(vae)
    image_size = int(vae.image_size)
    latent_tile_size = int(vae.latent_size)
    latent_size = int(volume_size) // factor
    image_overlap = int(tile_overlap) * factor

    image = prepare_anchor_image(
        anchor.image,
        num_phases=num_phases,
        segment=segment,
    )[0, 0].to(device=device)

    latent = torch.zeros(
        (int(vae.latent_ch), latent_size, latent_size),
        device=device,
        dtype=torch.float32,
    )
    count = torch.zeros((latent_size, latent_size), device=device, dtype=torch.float32)

    vae.eval()
    with torch.no_grad():
        for row, col in tile_grid(
            volume_size,
            volume_size,
            tile_size=image_size,
            overlap=image_overlap,
        ):
            patch = image[row : row + image_size, col : col + image_size].reshape(
                1,
                1,
                image_size,
                image_size,
            )
            mu, _ = vae.encode(patch)

            expected = torch.Size(
                [1, int(vae.latent_ch), latent_tile_size, latent_tile_size]
            )
            if mu.shape != expected:
                raise ValueError(
                    f"encoded anchor latent must have shape {tuple(expected)}."
                )

            require_finite("encoded anchor latent", mu)

            latent_row = row // factor
            latent_col = col // factor

            latent[
                :,
                latent_row : latent_row + latent_tile_size,
                latent_col : latent_col + latent_tile_size,
            ] += mu[0].detach()
            count[
                latent_row : latent_row + latent_tile_size,
                latent_col : latent_col + latent_tile_size,
            ] += 1

    return latent / count.clamp_min(1)


def _write_anchor_plane(
    latent: torch.Tensor,
    mask: torch.Tensor,
    encoded: torch.Tensor,
    *,
    axis: int,
    index: int,
    start: int,
) -> None:
    if axis == 0:
        latent[
            :,
            index,
            start : start + encoded.shape[1],
            start : start + encoded.shape[2],
        ] = encoded
        mask[
            :,
            index,
            start : start + encoded.shape[1],
            start : start + encoded.shape[2],
        ] = 1
        return

    if axis == 1:
        latent[
            :,
            start : start + encoded.shape[1],
            index,
            start : start + encoded.shape[2],
        ] = encoded
        mask[
            :,
            start : start + encoded.shape[1],
            index,
            start : start + encoded.shape[2],
        ] = 1
        return

    if axis == 2:
        latent[
            :,
            start : start + encoded.shape[1],
            start : start + encoded.shape[2],
            index,
        ] = encoded
        mask[
            :,
            start : start + encoded.shape[1],
            start : start + encoded.shape[2],
            index,
        ] = 1
        return

    raise ValueError("anchor axis must be 0, 1, or 2.")
