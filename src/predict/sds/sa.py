from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.predict.sds.phase import soft_phase_probability
from src.predict.sds.targets import phase_vector_target


def surface_area_loss(
    values: torch.Tensor,
    targets: Mapping[int, float] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    kernel_size: int = 7,
    sigma: float = 1.0,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")
    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_sa = compute_surface_area(
        values,
        num_phases=num_phases,
        temperature=temperature,
        kernel_size=kernel_size,
        sigma=sigma,
    )
    target_sa = phase_vector_target(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
        label="surface area value",
    )

    loss = weight * F.mse_loss(actual_sa, target_sa)
    stats = {
        "actual_sa": actual_sa.detach(),
        "target_sa": target_sa.detach(),
    }
    return loss, stats


def compute_surface_area(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    kernel_size: int = 7,
    sigma: float = 1.0,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")
    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    height, width = values.shape[-2:]
    slices = values.reshape(-1, height, width)
    probability = soft_phase_probability(
        slices,
        num_phases=num_phases,
        temperature=temperature,
        phase_dim=1,
    )
    return _relative_surface_area(
        probability,
        kernel_size=kernel_size,
        sigma=sigma,
    )


def _relative_surface_area(
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
    smooth = F.conv2d(
        probability,
        weight=kernel,
        padding=kernel_size // 2,
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
