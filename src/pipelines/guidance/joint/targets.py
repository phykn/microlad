from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.pipelines.guidance.joint.loss import phase_interface_matrices, texture_patches
from src.pipelines.guidance.metrics.runs import compute_run_profile


def reference_one_hot(
    reference_labels: torch.Tensor | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if reference_labels is None:
        raise ValueError("reference labels are required for joint reference guidance.")
    if not isinstance(reference_labels, torch.Tensor):
        raise TypeError("reference labels must be a tensor.")
    if reference_labels.ndim != 3 or reference_labels.shape[0] == 0:
        raise ValueError("reference labels must have shape [B, H, W].")
    if reference_labels.shape[-2:] != (image_size, image_size):
        raise ValueError("reference labels must match the joint volume slice size.")
    if reference_labels.dtype != torch.long:
        raise ValueError("reference labels must use torch.long dtype.")
    if reference_labels.min().item() < 0 or reference_labels.max().item() >= num_phases:
        raise ValueError("reference labels must contain valid phase labels.")
    labels = reference_labels.to(device=device)
    return F.one_hot(labels, num_classes=num_phases).permute(0, 3, 1, 2).to(dtype)


def texture_targets(
    real: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, int, int]] | None:
    if not enabled:
        return None
    if real is None:
        raise ValueError("reference labels are required for texture guidance.")
    targets = []
    for pool_size, kernel_size in ((1, 7), (2, 7), (4, 5)):
        patches = texture_patches(real, pool_size=pool_size, kernel_size=kernel_size)
        if patches.shape[0] > 2048:
            indices = torch.randperm(patches.shape[0], device=device)[:2048]
            patches = patches[indices]
        projection = F.normalize(
            torch.randn(patches.shape[1], 32, device=device, dtype=dtype),
            dim=0,
        )
        targets.append((patches.detach(), projection, pool_size, kernel_size))
    return targets


def interface_target(real: torch.Tensor | None, *, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    if real is None:
        raise ValueError("reference labels are required for interface guidance.")
    return phase_interface_matrices(real).mean(dim=0).detach()


def transition_target(real: torch.Tensor | None, *, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    if real is None:
        raise ValueError("reference labels are required for transition guidance.")
    horizontal = 1.0 - (real[:, :, :, :-1] * real[:, :, :, 1:]).sum(dim=1)
    vertical = 1.0 - (real[:, :, :-1, :] * real[:, :, 1:, :]).sum(dim=1)
    return 0.5 * (horizontal.mean() + vertical.mean())


def run_target(
    real: torch.Tensor | None,
    *,
    lengths: Sequence[int],
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    if not lengths:
        raise ValueError("run profile requires a volume size of at least 2.")
    if real is None:
        raise ValueError("reference labels are required for run-profile guidance.")
    return compute_run_profile(real, lengths=lengths).mean(dim=0).detach()
