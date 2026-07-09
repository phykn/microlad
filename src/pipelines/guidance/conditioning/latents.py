from collections.abc import Sequence

import torch

from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.validation import validate_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice


def prepare_anchor_latents(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    num_phases: int,
    segment: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not anchors:
        return None, None

    image_size = int(vae.image_size)
    validate_anchors(anchors, (image_size, image_size, image_size))

    latent_size = int(vae.latent_size)
    latent_ch = int(vae.latent_ch)
    shape = (latent_size, latent_ch, latent_size, latent_size)
    anchor_latent = torch.zeros(shape, device=device)
    anchor_mask = torch.zeros(shape, device=device)
    factor = _downsample_factor(vae)
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
            anchor_latent,
            anchor_mask,
            latent,
            axis=anchor.axis,
            index=latent_index,
        )

    return anchor_latent, anchor_mask


def _downsample_factor(vae: torch.nn.Module) -> int:
    return int(
        getattr(
            vae,
            "downsample_factor",
            int(vae.image_size) // int(vae.latent_size),
        )
    )


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
    anchor_latent: torch.Tensor,
    anchor_mask: torch.Tensor,
    latent: torch.Tensor,
    *,
    axis: int,
    index: int,
) -> None:
    mask = torch.ones_like(latent)

    if axis == 0:
        anchor_latent[index] = latent
        anchor_mask[index] = mask
        return

    plane = latent.permute(1, 0, 2).contiguous()
    plane_mask = mask.permute(1, 0, 2).contiguous()

    if axis == 1:
        anchor_latent[:, :, index, :] = plane
        anchor_mask[:, :, index, :] = plane_mask
    else:
        anchor_latent[:, :, :, index] = plane
        anchor_mask[:, :, :, index] = plane_mask
