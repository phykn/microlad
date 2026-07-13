from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.modeling.phases.relaxation import (
    calc_phase_probs,
    sharpen_phase_probabilities,
)
from src.pipelines.guidance.metrics.targets import build_phase_target
from src.validation import require_finite, require_finite_number, require_int


def interface_loss(
    values: torch.Tensor,
    targets: Mapping[int, float] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    kernel_size: int = 7,
    sigma: float = 1.0,
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

    require_int("kernel_size", kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    require_finite_number("sigma", sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    require_finite_number("weight", weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_interface = compute_interface_density(
        values,
        num_phases=num_phases,
        temperature=temperature,
        kernel_size=kernel_size,
        sigma=sigma,
        phase_probabilities=phase_probabilities,
    )
    target_interface = build_phase_target(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
        label="interface density",
    )

    loss = weight * F.mse_loss(actual_interface, target_interface)

    stats = {
        "actual_interface": actual_interface.detach(),
        "target_interface": target_interface.detach(),
    }

    return loss, stats


def compute_interface_density(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    kernel_size: int = 7,
    sigma: float = 1.0,
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

    require_int("kernel_size", kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    require_finite_number("sigma", sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    height, width = values.shape[-2:]
    slices = values.reshape(-1, height, width)
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

    return _interface_density(
        probability,
        kernel_size=kernel_size,
        sigma=sigma,
    )


def _interface_density(
    probability: torch.Tensor,
    *,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    _, num_phases, height, width = probability.shape
    kernel = _gaussian_kernel(
        kernel_size,
        sigma,
        device=probability.device,
        dtype=probability.dtype,
    ).repeat(num_phases, 1, 1, 1)
    padding = kernel_size // 2
    if padding > 0:
        probability = F.pad(
            probability,
            (padding, padding, padding, padding),
            mode="replicate",
        )

    smooth = F.conv2d(
        probability,
        weight=kernel,
        groups=num_phases,
    )

    tv_h = (smooth[:, :, 1:, :] - smooth[:, :, :-1, :]).abs().sum(dim=(2, 3))
    tv_w = (smooth[:, :, :, 1:] - smooth[:, :, :, :-1]).abs().sum(dim=(2, 3))

    return ((tv_h + tv_w) / (height * width)).mean(dim=0)


def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.arange(kernel_size, device=device, dtype=dtype)
    axis = axis - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, kernel_size, kernel_size)
