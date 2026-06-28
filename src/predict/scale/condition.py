from collections.abc import Sequence

import torch

from src.predict.anchor import prepare_anchor_image, validate_anchors
from src.predict.types import AnchorSlice


def center_start(*, volume_size: int, base_size: int) -> int:
    volume_size = int(volume_size)
    base_size = int(base_size)
    if volume_size < base_size:
        raise ValueError("volume_size must be at least base_size.")
    return (volume_size - base_size) // 2


def shifted_anchor_slices(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    base_size: int,
) -> list[tuple[int, int]]:
    start = center_start(volume_size=volume_size, base_size=base_size)
    return [(int(anchor.axis), start + int(anchor.index)) for anchor in anchors or []]


def prepare_scale_anchor_latents(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not anchors:
        return None, None

    factor = _downsample_factor(vae)
    if int(volume_size) % factor != 0:
        raise ValueError("volume_size must be divisible by VAE downsample factor.")

    latent_size = int(volume_size) // factor
    base_latent_size = int(vae.latent_size)
    start = center_start(volume_size=latent_size, base_size=base_latent_size)

    latent = torch.zeros(
        (int(vae.latent_ch), latent_size, latent_size, latent_size),
        device=device,
    )
    mask = torch.zeros_like(latent)

    for anchor in anchors:
        encoded = _encode_anchor(vae, anchor, num_phases=num_phases, segment=segment, device=device)
        index = start + int(anchor.index) // factor
        _write_anchor_plane(latent, mask, encoded, axis=int(anchor.axis), index=index, start=start)

    return latent, mask


def prepare_scale_anchor_targets(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    base_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    if not anchors:
        return {}, {}

    volume_size = int(volume_size)
    base_size = int(base_size)
    validate_anchors(anchors, (base_size, base_size, base_size))
    start = center_start(volume_size=volume_size, base_size=base_size)

    targets: dict[tuple[int, int], torch.Tensor] = {}
    masks: dict[tuple[int, int], torch.Tensor] = {}
    for anchor in anchors:
        image = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        )[0, 0].to(device=device, dtype=dtype)
        target = torch.zeros((volume_size, volume_size), device=device, dtype=dtype)
        mask = torch.zeros_like(target)
        target[start : start + base_size, start : start + base_size] = image
        mask[start : start + base_size, start : start + base_size] = 1
        targets[(int(anchor.axis), start + int(anchor.index))] = target
        masks[(int(anchor.axis), start + int(anchor.index))] = mask
    return targets, masks


def _downsample_factor(vae: torch.nn.Module) -> int:
    return int(
        getattr(
            vae,
            "downsample_factor",
            int(vae.image_size) // int(vae.latent_size),
        )
    )


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

    expected = torch.Size([1, int(vae.latent_ch), int(vae.latent_size), int(vae.latent_size)])
    if mu.shape != expected:
        raise ValueError(f"encoded anchor latent must have shape {tuple(expected)}.")
    return mu[0].detach()


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
        latent[:, index, start : start + encoded.shape[1], start : start + encoded.shape[2]] = encoded
        mask[:, index, start : start + encoded.shape[1], start : start + encoded.shape[2]] = 1
        return
    if axis == 1:
        latent[:, start : start + encoded.shape[1], index, start : start + encoded.shape[2]] = encoded
        mask[:, start : start + encoded.shape[1], index, start : start + encoded.shape[2]] = 1
        return
    if axis == 2:
        latent[:, start : start + encoded.shape[1], start : start + encoded.shape[2], index] = encoded
        mask[:, start : start + encoded.shape[1], start : start + encoded.shape[2], index] = 1
        return
    raise ValueError("anchor axis must be 0, 1, or 2.")
