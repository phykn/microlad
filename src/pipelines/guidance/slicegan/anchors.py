from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.modeling.slicegan import MIN_SLICE_SIZE, SCALE_FACTOR
from src.modeling.vae import get_downsample_factor
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.scaling.tiles import tile_grid
from src.validation import require_finite


@dataclass(frozen=True)
class AnchorPatch:
    labels: torch.Tensor
    probabilities: torch.Tensor
    latent: torch.Tensor
    axis: int
    latent_index: int
    latent_row: int
    latent_col: int


@dataclass(frozen=True)
class PreparedAnchor:
    labels: torch.Tensor
    axis: int
    index: int
    start: int
    patches: tuple[AnchorPatch, ...]


def validate_inputs(
    vae: torch.nn.Module,
    *,
    anchors: Sequence[VolumeAnchor],
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    volume_size: int,
    num_phases: int,
) -> tuple[int, int]:
    if not anchors:
        raise ValueError("conditional SliceGAN requires at least one anchor.")
    if getattr(vae, "num_phases", None) != num_phases:
        raise ValueError("num_phases must match the categorical VAE.")
    if not callable(getattr(vae, "encode", None)) or not callable(
        getattr(vae, "decode_probs", None)
    ):
        raise ValueError("SliceGAN requires categorical VAE encode and decode_probs.")
    factor = get_downsample_factor(vae)
    if volume_size < int(vae.image_size) or volume_size % factor != 0:
        raise ValueError(
            "volume_size must be at least vae.image_size and divisible by its downsample factor."
        )
    base_latent = int(vae.latent_size)
    output_latent = volume_size // factor
    for name, size in (
        ("vae.latent_size", base_latent),
        ("output latent size", output_latent),
    ):
        if size < MIN_SLICE_SIZE or size % SCALE_FACTOR != 0:
            raise ValueError(
                f"{name} must be at least {MIN_SLICE_SIZE} and divisible by {SCALE_FACTOR}."
            )
    if not 0.0 <= phase_fraction_tolerance <= 1.0:
        raise ValueError("phase_fraction_tolerance must be between 0 and 1.")
    if target_fraction is not None:
        fractions = torch.as_tensor(target_fraction)
        if fractions.shape != (num_phases,):
            raise ValueError("target_fraction must have shape [num_phases].")
        if not torch.isfinite(fractions).all() or torch.any(fractions < 0.0):
            raise ValueError("target_fraction must contain finite non-negative values.")
        if not torch.allclose(
            fractions.sum(),
            torch.ones_like(fractions.sum()),
            atol=1e-4,
        ):
            raise ValueError("target_fraction must sum to one.")
    for anchor in anchors:
        labels = anchor.image
        if labels.ndim != 2 or labels.shape[0] != labels.shape[1]:
            raise ValueError("anchor images must be square categorical slices.")
        if not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()):
            raise ValueError("anchor labels must contain finite integer phase values.")
        size = int(labels.shape[-1])
        if size < int(vae.image_size) or size > volume_size:
            raise ValueError("anchor size must be between vae.image_size and volume_size.")
        if size % factor != 0 or int(anchor.start) % factor != 0:
            raise ValueError("anchor size and start must align to the VAE latent grid.")
        if anchor.axis not in (0, 1, 2):
            raise ValueError("anchor axis must be 0, 1, or 2.")
        if anchor.index < 0 or anchor.index >= volume_size:
            raise ValueError("anchor index must be inside volume_size.")
        if anchor.start < 0 or anchor.start + size > volume_size:
            raise ValueError("anchor image must fit inside the selected volume slice.")
        if labels.min().item() < 0 or labels.max().item() >= num_phases:
            raise ValueError("anchor labels must be inside the phase range.")
    return factor, output_latent


def prepare_anchors(
    vae: torch.nn.Module,
    anchors: Sequence[VolumeAnchor],
    *,
    factor: int,
    volume_size: int,
    num_phases: int,
    device: torch.device,
) -> list[PreparedAnchor]:
    prepared = []
    image_size = int(vae.image_size)
    latent_size = int(vae.latent_size)
    output_latent_size = volume_size // factor
    for anchor in anchors:
        labels = anchor.image.to(device=device, dtype=torch.long)
        patches = []
        overlap = image_size // 4 if int(labels.shape[-1]) > image_size else 0
        for row, col in tile_grid(
            int(labels.shape[-2]),
            int(labels.shape[-1]),
            tile_size=image_size,
            overlap=overlap,
        ):
            patch_labels = labels[row : row + image_size, col : col + image_size]
            image = patch_labels.reshape(1, 1, image_size, image_size).float()
            with torch.no_grad():
                latent, _ = vae.encode(image)
            expected = (1, int(vae.latent_ch), latent_size, latent_size)
            if tuple(latent.shape) != expected:
                raise ValueError(f"encoded anchor latent must have shape {expected}.")
            require_finite("encoded anchor latent", latent)
            absolute_row = int(anchor.start) + row
            absolute_col = int(anchor.start) + col
            if absolute_row % factor != 0 or absolute_col % factor != 0:
                raise ValueError("anchor tiles must align to the VAE latent grid.")
            patches.append(
                AnchorPatch(
                    labels=patch_labels,
                    probabilities=F.one_hot(
                        patch_labels,
                        num_classes=num_phases,
                    ).movedim(-1, 0).float(),
                    latent=latent[0].detach(),
                    axis=int(anchor.axis),
                    latent_index=min(
                        int(anchor.index) // factor,
                        output_latent_size - 1,
                    ),
                    latent_row=absolute_row // factor,
                    latent_col=absolute_col // factor,
                )
            )
        prepared.append(
            PreparedAnchor(
                labels=labels,
                axis=int(anchor.axis),
                index=int(anchor.index),
                start=int(anchor.start),
                patches=tuple(patches),
            )
        )
    return prepared


def anchor_mismatches(
    volume: torch.Tensor,
    anchors: Sequence[PreparedAnchor],
) -> torch.Tensor:
    values = []
    for anchor in anchors:
        plane = volume.select(anchor.axis, anchor.index)
        size = int(anchor.labels.shape[-1])
        actual = plane[
            anchor.start : anchor.start + size,
            anchor.start : anchor.start + size,
        ]
        values.append((actual != anchor.labels).float().mean())
    return torch.stack(values)


def local_boundary_stats(
    volume: torch.Tensor,
    anchor: PreparedAnchor,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = int(volume.shape[anchor.axis])
    changes = (
        volume.narrow(anchor.axis, 1, length - 1)
        != volume.narrow(anchor.axis, 0, length - 1)
    ).float()
    size = int(anchor.labels.shape[-1])
    region = [slice(None)] * 3
    for axis in range(3):
        if axis != anchor.axis:
            region[axis] = slice(anchor.start, anchor.start + size)
    changes = changes[tuple(region)]
    profile = changes.mean(
        dim=tuple(axis for axis in range(3) if axis != anchor.axis)
    )
    start = max(0, anchor.index - 5)
    stop = min(int(profile.shape[0]), anchor.index + 5)
    local = profile[start:stop]
    jump = (
        (local[1:] - local[:-1]).abs().max()
        if local.numel() > 1
        else local.new_zeros(())
    )
    return local.std(unbiased=False), jump
