import torch
import torch.nn.functional as F

from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.modeling.phases.representation import (
    geometric_probability_consensus,
    probabilities_to_relaxed_labels,
)
from src.modeling.vae import get_downsample_factor
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def generate_initial_volume(
    sampler,
    vae: torch.nn.Module,
    *,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    axis_consensus: bool = False,
) -> torch.Tensor:
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
    return decode_volume(vae, latent)


def decode_latent(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError("latent must have shape [1, C, H, W].")

    values, probabilities = decode_latents(
        vae,
        latent,
        num_phases=num_phases,
    )
    return values[0], probabilities[0]


def decode_latents(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_latents(vae, latents)
    _validate_decoder(vae, num_phases)

    probabilities = vae.decode_probs(latents)
    expected_shape = (
        latents.shape[0],
        num_phases,
        int(vae.image_size),
        int(vae.image_size),
    )
    if probabilities.shape != expected_shape:
        raise ValueError("vae.decode_probs must return shape [B, num_phases, H, W].")
    require_finite("decoded probabilities", probabilities)

    values = probabilities_to_relaxed_labels(probabilities, num_phases)[:, 0]
    return values.float(), probabilities.float()


@torch.no_grad()
def decode_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    probabilities = decode_volume_probs(vae, latent)
    num_phases = int(vae.num_phases)
    return probabilities_to_calibrated_labels(
        probabilities,
        num_phases,
    )[0, 0].float()


@torch.no_grad()
def decode_volume_probs(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    num_phases: int | None = None,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    if num_phases is None:
        num_phases = getattr(vae, "num_phases", None)
    _validate_decoder(vae, num_phases)

    output_size = int(latent.shape[1]) * get_downsample_factor(vae)
    latent_batches = (
        latent.permute(1, 0, 2, 3).contiguous(),
        latent.permute(2, 0, 1, 3).contiguous(),
        latent.permute(3, 0, 1, 2).contiguous(),
    )
    probability_planes = [
        decode_latents(vae, batch, num_phases=num_phases)[1]
        for batch in latent_batches
    ]

    depth_probs = interpolate_phase_planes(
        probability_planes[0],
        output_size=output_size,
    )
    height_probs = interpolate_phase_planes(
        probability_planes[1],
        output_size=output_size,
    ).permute(0, 2, 1, 3)
    width_probs = interpolate_phase_planes(
        probability_planes[2],
        output_size=output_size,
    ).permute(0, 2, 3, 1)

    return geometric_probability_consensus(
        torch.stack([depth_probs, height_probs, width_probs]),
        num_phases,
    ).unsqueeze(0)


def interpolate_phase_planes(
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


def interpolate_planes(planes: torch.Tensor, output_size: int) -> torch.Tensor:
    return F.interpolate(
        planes.unsqueeze(0).unsqueeze(0),
        size=(output_size, planes.shape[1], planes.shape[2]),
        mode="trilinear",
        align_corners=False,
    )[0, 0]


def _validate_decoder(vae: torch.nn.Module, num_phases: int) -> None:
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if num_phases != getattr(vae, "num_phases", None):
        raise ValueError("num_phases must match vae.num_phases.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise ValueError("categorical decoding requires vae.decode_probs.")


def _validate_latents(vae: torch.nn.Module, latents: torch.Tensor) -> None:
    if latents.ndim != 4:
        raise ValueError("latents must have shape [B, C, H, W].")
    if latents.shape[0] <= 0:
        raise ValueError("latents must contain at least one sample.")
    if latents.shape[1] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")
    latent_size = int(vae.latent_size)
    if latents.shape[2:] != (latent_size, latent_size):
        raise ValueError("latent spatial shape must match vae.latent_size.")
    require_float("latent dtype", latents.dtype)
    require_finite("latent", latents)


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
