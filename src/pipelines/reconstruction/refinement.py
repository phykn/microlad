import torch

from src.common.tensors.validation import require_finite, require_float
from src.modeling.phases.representation import (
    geometric_probability_consensus,
    probabilities_to_calibrated_labels,
)


@torch.no_grad()
def refine_axes(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
) -> torch.Tensor:
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    _validate_volume(volume, vae)

    refined = volume.float()

    if steps == 0:
        return refined

    vae.eval()

    for _ in range(steps):
        refined = _refine_once(refined, vae)

    return refined


def _refine_once(volume: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    num_phases = getattr(vae, "num_phases", None)
    if (
        isinstance(num_phases, int)
        and not isinstance(num_phases, bool)
        and callable(getattr(vae, "decode_probs", None))
    ):
        return _refine_categorical_once(volume, vae, num_phases=num_phases)

    depth, height, width = volume.shape
    new_volume = torch.zeros_like(volume)

    decoded = _encode_decode_batch(
        vae,
        volume.reshape(depth, 1, height, width),
    )
    new_volume += decoded[:, 0, :, :]

    decoded = _encode_decode_batch(
        vae,
        volume.permute(1, 0, 2).contiguous().view(height, 1, depth, width),
    )
    new_volume += decoded[:, 0, :, :].permute(1, 0, 2)

    decoded = _encode_decode_batch(
        vae,
        volume.permute(2, 0, 1).contiguous().view(width, 1, depth, height),
    )
    new_volume += decoded[:, 0, :, :].permute(1, 2, 0)

    return (new_volume / 3.0).float()


def _refine_categorical_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    num_phases: int,
) -> torch.Tensor:
    depth, height, width = volume.shape

    depth_probs = _encode_decode_probabilities(
        vae,
        volume.reshape(depth, 1, height, width),
        num_phases=num_phases,
    ).permute(1, 0, 2, 3)
    height_probs = _encode_decode_probabilities(
        vae,
        volume.permute(1, 0, 2).contiguous().view(height, 1, depth, width),
        num_phases=num_phases,
    ).permute(1, 2, 0, 3)
    width_probs = _encode_decode_probabilities(
        vae,
        volume.permute(2, 0, 1).contiguous().view(width, 1, depth, height),
        num_phases=num_phases,
    ).permute(1, 2, 3, 0)

    axis_probabilities = torch.stack(
        [depth_probs, height_probs, width_probs],
        dim=0,
    )
    probabilities = geometric_probability_consensus(
        axis_probabilities,
        num_phases,
    ).unsqueeze(0)
    return probabilities_to_calibrated_labels(probabilities, num_phases)[0, 0].float()


def _encode_decode_probabilities(
    vae: torch.nn.Module,
    images: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    mu, _ = vae.encode(images)
    if mu.ndim != 4 or mu.shape[0] != images.shape[0]:
        raise ValueError("encode output must have shape [B, C, H, W].")
    require_finite("encoded latent", mu)

    probabilities = vae.decode_probs(mu)
    expected_shape = (images.shape[0], num_phases, *images.shape[-2:])
    if probabilities.shape != expected_shape:
        raise ValueError(
            "decode_probs output must have shape [B, num_phases, H, W]."
        )
    require_finite("decoded probabilities", probabilities)
    return probabilities.float()


def _encode_decode_batch(vae: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape [B, 1, H, W].")

    mu, _ = vae.encode(images)

    if mu.ndim != 4:
        raise ValueError("encode output must have shape [B, C, H, W].")

    if mu.shape[0] != images.shape[0]:
        raise ValueError("encode output batch size must match the input batch.")

    require_finite("encoded latent", mu)

    decoded = vae.decode(mu)

    if decoded.ndim != 4 or decoded.shape[:2] != (images.shape[0], 1):
        raise ValueError("decode output must have shape [B, 1, H, W].")

    if decoded.shape[-2:] != images.shape[-2:]:
        raise ValueError("decode output spatial shape must match the input slice.")

    require_finite("decoded slice", decoded)

    return decoded.float()


def _validate_volume(volume: torch.Tensor, vae: torch.nn.Module) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    require_float("volume dtype", volume.dtype)
    require_finite("volume", volume)

    depth, height, width = volume.shape

    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("three-axis refinement requires a cubic volume.")

    if depth != int(vae.image_size):
        raise ValueError("volume size must match vae.image_size.")
