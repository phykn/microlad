from collections.abc import Mapping
from numbers import Integral

import torch
import torch.nn.functional as F

from src.modeling.phases.relaxation import (
    calc_phase_probs,
    sharpen_phase_probabilities,
)
from src.validation import require_finite, require_finite_number, require_int


def correlation_loss(
    values: torch.Tensor,
    targets: Mapping[int, torch.Tensor] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
    phase_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")

    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")

    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    require_finite_number("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    require_finite_number("weight", weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_correlation = compute_correlation(
        values,
        num_phases=num_phases,
        temperature=temperature,
        phase_probabilities=phase_probabilities,
    )
    target_correlation = _target_tensor(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
    )

    if target_correlation.shape[1] != actual_correlation.shape[1]:
        raise ValueError("targets length must match the predicted correlation length.")

    loss = weight * F.mse_loss(actual_correlation, target_correlation)

    stats = {
        "actual_correlation": actual_correlation.detach(),
        "target_correlation": target_correlation.detach(),
    }

    return loss, stats


def compute_correlation(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    phase_probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")

    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")

    require_finite("values", values)
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    require_finite_number("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    height, width = values.shape[-2:]
    slices = values.reshape(-1, height, width)
    bin_matrix, bin_counts = _build_tpc_bins(
        height,
        width,
        device=values.device,
        dtype=values.dtype,
    )

    if phase_probabilities is None:
        probability = calc_phase_probs(
            slices,
            num_phases=num_phases,
            temperature=temperature,
            phase_dim=1,
        )
    else:
        probability = sharpen_phase_probabilities(
            phase_probabilities,
            num_phases=num_phases,
            temperature=temperature,
        )

    return _phase_correlation(probability, bin_matrix, bin_counts)


def _phase_correlation(
    probability: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    phase_profiles = []

    for phase in range(probability.shape[1]):
        correlation = _compute_correlation_batch(
            probability[:, phase],
            bin_matrix,
            bin_counts,
        )
        phase_profiles.append(correlation.mean(dim=0))

    return torch.stack(phase_profiles, dim=0)


def _compute_correlation_batch(
    masks: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    height, width = masks.shape[-2:]
    fft = torch.fft.fft2(masks, dim=(-2, -1))
    corr = torch.fft.ifft2(fft * torch.conj(fft), dim=(-2, -1)) / (height * width)
    corr = torch.real(torch.fft.fftshift(corr, dim=(-2, -1))).reshape(masks.shape[0], -1)
    return (corr @ bin_matrix.transpose(0, 1)) / bin_counts.squeeze(1).clamp_min(1)


def _build_tpc_bins(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if height <= 0 or width <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    radius = (
        ((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
        .sqrt()
        .round()
        .long()
        .view(-1)
    )
    num_bins = int(radius.max().item()) + 1
    bin_matrix = F.one_hot(radius, num_classes=num_bins).to(dtype=dtype).transpose(0, 1)
    bin_counts = bin_matrix.sum(dim=1, keepdim=True)
    return bin_matrix, bin_counts


def _target_tensor(
    targets: Mapping[int, torch.Tensor] | torch.Tensor,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        target = targets.to(device=device, dtype=dtype)
    else:
        expected_keys = set(range(num_phases))
        profiles: dict[int, torch.Tensor] = {}

        for phase, profile in targets.items():
            if not isinstance(phase, Integral) or isinstance(phase, bool):
                raise ValueError("targets phase indices must be integers.")

            phase = int(phase)
            if phase < 0 or phase >= num_phases:
                raise ValueError("targets must contain phase indices within num_phases.")

            profiles[phase] = torch.as_tensor(profile, device=device, dtype=dtype)

        if set(profiles) != expected_keys:
            raise ValueError("targets must contain one correlation profile per phase.")

        target = torch.stack([profiles[phase] for phase in range(num_phases)], dim=0)

    if target.ndim != 2 or target.shape[0] != num_phases:
        raise ValueError("targets must have shape [num_phases, num_bins].")

    if target.shape[1] < 1:
        raise ValueError("targets must contain at least one correlation bin.")

    require_finite("targets", target)

    if torch.any(target < 0):
        raise ValueError("targets must be non-negative.")

    if torch.any(target > 1):
        raise ValueError("targets values must be between 0 and 1.")

    return target
