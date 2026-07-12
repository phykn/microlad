from collections.abc import Sequence
import math

import torch

from src.common.validation import require_finite_number
from src.modeling.vae import get_downsample_factor
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.validation import validate_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice


def encode_anchors(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    num_phases: int,
    segment: bool,
    device: torch.device,
    spread_sigma: float = 0.0,
    peak_strength: float = 1.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not anchors:
        return None, None

    image_size = int(vae.image_size)
    validate_anchors(anchors, (image_size, image_size, image_size))
    require_finite_number("spread_sigma", spread_sigma)
    require_finite_number("peak_strength", peak_strength)
    if spread_sigma < 0.0:
        raise ValueError("spread_sigma must be non-negative.")
    if peak_strength <= 0.0 or peak_strength > 1.0:
        raise ValueError("peak_strength must be greater than 0 and at most 1.")

    latent_size = int(vae.latent_size)
    latent_ch = int(vae.latent_ch)
    shape = (latent_size, latent_ch, latent_size, latent_size)
    latent_sum = torch.zeros(shape, device=device)
    weight_sum = torch.zeros(shape, device=device)
    factor = get_downsample_factor(vae)
    written_planes: set[tuple[int, int]] = set()

    for anchor in anchors:
        latent_index = min(anchor.index // factor, latent_size - 1)
        plane_key = (int(anchor.axis), int(latent_index))

        if plane_key in written_planes:
            raise ValueError("anchor slices collapse to the same latent plane.")

        written_planes.add(plane_key)

        latent = _encode_anchor_latent(
            vae,
            anchor,
            num_phases=num_phases,
            segment=segment,
            device=device,
        )

        _write_anchor_latent(
            latent_sum,
            weight_sum,
            latent,
            axis=anchor.axis,
            index=latent_index,
            spread_sigma=spread_sigma,
            peak_strength=peak_strength,
        )

    active = weight_sum > 0.0
    anchor_latent = torch.zeros_like(latent_sum)
    anchor_latent[active] = latent_sum[active] / weight_sum[active]
    anchor_mask = weight_sum.clamp(max=1.0)
    return anchor_latent, anchor_mask


def _encode_anchor_latent(
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
        [
            1,
            int(vae.latent_ch),
            int(vae.latent_size),
            int(vae.latent_size),
        ]
    )
    if mu.shape != expected:
        raise ValueError(f"encoded anchor latent must have shape {tuple(expected)}.")
    return mu[0].detach()


def _write_anchor_latent(
    latent_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    latent: torch.Tensor,
    *,
    axis: int,
    index: int,
    spread_sigma: float,
    peak_strength: float,
) -> None:
    size = int(latent_sum.shape[0])
    propagated = _propagate_anchor_latent(
        latent,
        size=size,
        index=index,
        spread_sigma=spread_sigma,
    )
    for plane_index in range(size):
        distance = abs(plane_index - index)
        if spread_sigma == 0.0:
            if distance != 0:
                continue
            weight = peak_strength
        else:
            weight = peak_strength * math.exp(
                -0.5 * (distance / spread_sigma) ** 2
            )
        latent_plane = propagated[plane_index]
        oriented_plane = latent_plane.permute(1, 0, 2).contiguous()

        if axis == 0:
            latent_sum[plane_index].add_(latent_plane, alpha=weight)
            weight_sum[plane_index].add_(weight)
        elif axis == 1:
            latent_sum[:, :, plane_index, :].add_(oriented_plane, alpha=weight)
            weight_sum[:, :, plane_index, :].add_(weight)
        else:
            latent_sum[:, :, :, plane_index].add_(oriented_plane, alpha=weight)
            weight_sum[:, :, :, plane_index].add_(weight)


def _propagate_anchor_latent(
    latent: torch.Tensor,
    *,
    size: int,
    index: int,
    spread_sigma: float,
) -> list[torch.Tensor]:
    planes = [latent] * size
    if spread_sigma == 0.0:
        return planes

    correlation_length = max(2.0 * spread_sigma, 1.0)
    rho = math.exp(-0.5 / correlation_length**2)
    innovation_scale = math.sqrt(max(0.0, 1.0 - rho**2))
    planes[index] = latent
    for direction in (-1, 1):
        current = latent
        plane_index = index + direction
        while 0 <= plane_index < size:
            current = (
                rho * current
                + innovation_scale * torch.randn_like(current)
            )
            planes[plane_index] = current
            plane_index += direction
    return planes
