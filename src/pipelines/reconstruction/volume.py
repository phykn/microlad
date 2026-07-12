import torch
import torch.nn.functional as F

from src.common.validation import require_int
from src.common.tensors.validation import require_finite, require_float
from src.modeling.phases.representation import (
    geometric_probability_consensus,
    probabilities_to_calibrated_labels,
    probabilities_to_relaxed_labels,
)
from src.modeling.vae import get_downsample_factor


@torch.no_grad()
def generate_initial_volume(
    sampler,
    vae: torch.nn.Module,
    *,
    size: int | None = None,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    axis_consensus: bool = False,
) -> torch.Tensor:
    if size is None:
        size = int(vae.image_size)

    require_int("size", size)

    if size != int(vae.image_size):
        raise ValueError("size must match vae.image_size.")

    latent_ch = int(vae.latent_ch)
    latent_size = int(vae.latent_size)
    vae.eval()

    latent_batch = sampler.sample_lmpdd(
        (latent_size, latent_ch, latent_size, latent_size),
        anchor_latent=anchor_latent,
        anchor_mask=anchor_mask,
        axis_consensus=axis_consensus,
    )
    latent = latent_batch.permute(1, 0, 2, 3).contiguous()
    return decode_latent_volume(vae, latent)


def decode_latent(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latent)

    if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
        raise ValueError("vae.decode must return shape [1, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    require_finite("decoded", decoded)

    return decoded[0, 0]


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents)

    if (
        decoded.ndim != 4
        or decoded.shape[0] != latents.shape[0]
        or decoded.shape[1] != 1
    ):
        raise ValueError("vae.decode must return shape [B, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    require_finite("decoded", decoded)

    return decoded[:, 0]


def decode_latents_with_probabilities(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    decode_probs = getattr(vae, "decode_probs", None)
    vae_num_phases = getattr(vae, "num_phases", None)

    if callable(decode_probs) and vae_num_phases == num_phases:
        probabilities = decode_probs(latents)
        expected_shape = (
            latents.shape[0],
            num_phases,
            int(vae.image_size),
            int(vae.image_size),
        )
        if probabilities.shape != expected_shape:
            raise ValueError(
                "vae.decode_probs must return shape [B, num_phases, H, W]."
            )
        require_finite("decoded probabilities", probabilities)
        values = probabilities_to_relaxed_labels(probabilities, num_phases)[:, 0]
        return values.float(), probabilities.float()

    return decode_latents(vae, latents), None


def decode_latent_with_probabilities(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    values, probabilities = decode_latents_with_probabilities(
        vae,
        latent,
        num_phases=num_phases,
    )
    if values.shape[0] != 1:
        raise ValueError("latent batch size must be one.")
    return values[0], None if probabilities is None else probabilities[0]


def decoded_labels(
    values: torch.Tensor,
    probabilities: torch.Tensor | None,
    *,
    num_phases: int,
) -> torch.Tensor:
    if probabilities is None:
        return values

    return probabilities_to_calibrated_labels(probabilities, num_phases)[:, 0].float()


@torch.no_grad()
def decode_latent_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    _, depth, height, width = latent.shape
    factor = get_downsample_factor(vae)
    output_size = depth * factor

    num_phases = getattr(vae, "num_phases", None)
    if (
        isinstance(num_phases, int)
        and not isinstance(num_phases, bool)
        and callable(getattr(vae, "decode_probs", None))
    ):
        return _decode_categorical_volume(
            vae,
            latent,
            output_size=output_size,
            num_phases=num_phases,
        )

    depth_planes = torch.stack(
        [
            decode_latent(vae, latent[:, d, :, :].unsqueeze(0)).float()
            for d in range(depth)
        ]
    )
    depth_volume = _interpolate_planes(depth_planes, output_size)

    height_planes = torch.stack(
        [
            decode_latent(vae, latent[:, :, h, :].unsqueeze(0)).float()
            for h in range(height)
        ]
    )
    height_volume = _interpolate_planes(height_planes, output_size).permute(1, 0, 2)

    width_planes = torch.stack(
        [
            decode_latent(vae, latent[:, :, :, w].unsqueeze(0)).float()
            for w in range(width)
        ]
    )

    width_volume = _interpolate_planes(width_planes, output_size).permute(1, 2, 0)

    return ((depth_volume + height_volume + width_volume) / 3.0).float()


def _decode_categorical_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    output_size: int,
    num_phases: int,
) -> torch.Tensor:
    latent_batches = (
        latent.permute(1, 0, 2, 3).contiguous(),
        latent.permute(2, 0, 1, 3).contiguous(),
        latent.permute(3, 0, 1, 2).contiguous(),
    )
    probability_planes = []
    for latent_batch in latent_batches:
        _, probabilities = decode_latents_with_probabilities(
            vae,
            latent_batch,
            num_phases=num_phases,
        )
        if probabilities is None:
            raise ValueError("categorical volume decoding requires phase probabilities.")
        probability_planes.append(probabilities)

    depth_probs = _interpolate_phase_planes(
        probability_planes[0],
        output_size=output_size,
    )
    height_probs = _interpolate_phase_planes(
        probability_planes[1],
        output_size=output_size,
    ).permute(0, 2, 1, 3)
    width_probs = _interpolate_phase_planes(
        probability_planes[2],
        output_size=output_size,
    ).permute(0, 2, 3, 1)

    axis_probabilities = torch.stack(
        [depth_probs, height_probs, width_probs],
        dim=0,
    )
    fused = geometric_probability_consensus(
        axis_probabilities,
        num_phases,
    ).unsqueeze(0)
    return probabilities_to_calibrated_labels(fused, num_phases)[0, 0].float()


def _interpolate_phase_planes(
    probabilities: torch.Tensor,
    *,
    output_size: int,
) -> torch.Tensor:
    return F.interpolate(
        probabilities.permute(1, 0, 2, 3).unsqueeze(0),
        size=(output_size, probabilities.shape[2], probabilities.shape[3]),
        mode="trilinear",
        align_corners=False,
    )[0]


def _interpolate_planes(planes: torch.Tensor, output_size: int) -> torch.Tensor:
    return F.interpolate(
        planes.unsqueeze(0).unsqueeze(0),
        size=(output_size, planes.shape[1], planes.shape[2]),
        mode="trilinear",
        align_corners=False,
    )[0, 0]


def _validate_latent_volume(vae: torch.nn.Module, latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise ValueError("latent volume must have shape [C, D, H, W].")

    if latent.shape[0] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")

    require_float("latent dtype", latent.dtype)
    require_finite("latent", latent)

    latent_size = int(vae.latent_size)

    if latent.shape[1:] != (latent_size, latent_size, latent_size):
        raise ValueError(
            f"latent spatial shape must be {(latent_size, latent_size, latent_size)}."
        )
