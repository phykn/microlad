from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.pipelines.guidance.descriptors.run_profile import compute_run_profile
from src.pipelines.guidance.descriptors.topology import compute_euler_density
from src.pipelines.guidance.slices import volume_slices


def phase_volume_diagnostics(
    volume: torch.Tensor,
    references: torch.Tensor,
    *,
    target_fraction: torch.Tensor,
    num_phases: int,
    run_lengths: Sequence[int] = (2, 4, 8, 16),
) -> dict[str, torch.Tensor]:
    """Measure continuity, repetition, morphology, and phase-fraction errors."""

    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if any(int(size) < 2 for size in volume.shape):
        raise ValueError("volume dimensions must contain at least two voxels.")
    labels = volume.to(dtype=torch.long)
    if labels.min().item() < 0 or labels.max().item() >= num_phases:
        raise ValueError("volume labels must be inside the phase range.")
    probabilities = (
        F.one_hot(labels, num_classes=num_phases).movedim(-1, 0).unsqueeze(0).float()
    )
    references = references.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    if references.ndim != 4 or references.shape[1] != num_phases:
        raise ValueError("references must have shape [B, P, H, W].")

    axis_run_profile = compute_run_profile(probabilities, lengths=run_lengths)
    target_run_profile = compute_run_profile(
        references,
        lengths=run_lengths,
    ).mean(dim=0)
    axis_run_profile_mae = torch.mean(
        torch.abs(axis_run_profile - target_run_profile.unsqueeze(0)),
        dim=(1, 2),
    )
    target_euler_density = compute_euler_density(references).mean(dim=0)

    transition_rates = []
    lag3_change_rates = []
    exact_repeat_rates = []
    global_boundary_jumps = []
    axis_euler_density = []
    for axis in range(3):
        length = int(labels.shape[axis])
        before = labels.narrow(axis, 0, length - 1)
        after = labels.narrow(axis, 1, length - 1)
        changed = (before != after).float()
        reduce_axes = tuple(dimension for dimension in range(3) if dimension != axis)
        boundary_profile = changed.mean(dim=reduce_axes)
        transition_rates.append(boundary_profile.mean())
        global_boundary_jumps.append(
            torch.abs(boundary_profile[1:] - boundary_profile[:-1]).max()
            if boundary_profile.numel() > 1
            else boundary_profile.new_zeros(())
        )
        same_slices = (before == after).movedim(axis, 0).reshape(length - 1, -1)
        exact_repeat_rates.append(same_slices.all(dim=1).float().mean())
        lag = min(3, length - 1)
        lag3_change_rates.append(
            (
                labels.narrow(axis, 0, length - lag)
                != labels.narrow(axis, lag, length - lag)
            )
            .float()
            .mean()
        )
        axis_euler_density.append(
            compute_euler_density(
                volume_slices(probabilities, axis, num_phases=num_phases)
            ).mean(dim=0)
        )

    axis_euler_density = torch.stack(axis_euler_density)
    measured_fraction = probabilities.mean(dim=(0, 2, 3, 4))
    target_fraction = torch.as_tensor(
        target_fraction,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return {
        "axis_transition_rate": torch.stack(transition_rates),
        "axis_lag3_change_rate": torch.stack(lag3_change_rates),
        "axis_exact_repeat_rate": torch.stack(exact_repeat_rates),
        "axis_global_boundary_jump": torch.stack(global_boundary_jumps),
        "axis_run_profile": axis_run_profile,
        "target_run_profile": target_run_profile,
        "axis_run_profile_mae": axis_run_profile_mae,
        "axis_euler_density": axis_euler_density,
        "target_euler_density": target_euler_density,
        "axis_euler_mae": torch.mean(
            torch.abs(axis_euler_density - target_euler_density.unsqueeze(0)),
            dim=1,
        ),
        "phase_fraction_error": measured_fraction - target_fraction,
    }
