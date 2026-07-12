from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.common.tensors.validation import require_finite
from src.common.validation import require_finite_number, require_int


def compute_run_profile(
    probabilities: torch.Tensor,
    *,
    lengths: Sequence[int],
) -> torch.Tensor:
    """Measure normalized same-phase run survival along every spatial axis."""
    run_lengths = _validate_inputs(probabilities, lengths)
    spatial_dimensions = tuple(range(2, probabilities.ndim))
    occupancy = probabilities.mean(dim=(0, *spatial_dimensions))
    tiny = torch.finfo(probabilities.dtype).tiny
    direction_profiles = []

    for dimension in spatial_dimensions:
        oriented = probabilities.movedim(dimension, -1)
        line_length = int(oriented.shape[-1])
        other_shape = tuple(int(size) for size in oriented.shape[2:-1])
        lines = oriented.reshape(-1, 1, line_length)
        length_profiles = []

        for run_length in run_lengths:
            pooled = F.avg_pool1d(
                lines,
                kernel_size=run_length,
                stride=1,
            )
            pooled = pooled.reshape(
                probabilities.shape[0],
                probabilities.shape[1],
                *other_shape,
                line_length - run_length + 1,
            )
            reduce_dimensions = tuple(
                index for index in range(pooled.ndim) if index != 1
            )
            survival = pooled.pow(run_length).mean(dim=reduce_dimensions)
            length_profiles.append(
                (survival / occupancy.clamp_min(tiny)).clamp(0.0, 1.0)
            )

        direction_profiles.append(torch.stack(length_profiles, dim=-1))

    return torch.stack(direction_profiles, dim=0)


def run_profile_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    lengths: Sequence[int],
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    require_finite_number("weight", weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual = compute_run_profile(probabilities, lengths=lengths)
    target = torch.as_tensor(
        targets,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected_shape = actual.shape[1:]
    if target.shape != expected_shape:
        raise ValueError(
            "targets must have shape [num_phases, num_run_lengths]."
        )
    require_finite("run profile targets", target)
    if torch.any(target < 0.0) or torch.any(target > 1.0):
        raise ValueError("run profile targets must be between 0 and 1.")

    expected = target.unsqueeze(0).expand_as(actual)
    loss = weight * F.mse_loss(actual, expected)
    return loss, {
        "actual_run_profile": actual.detach(),
        "target_run_profile": target.detach(),
    }


def _validate_inputs(
    probabilities: torch.Tensor,
    lengths: Sequence[int],
) -> tuple[int, ...]:
    if probabilities.ndim not in (4, 5):
        raise ValueError(
            "probabilities must have shape [B, P, H, W] or [B, P, D, H, W]."
        )
    if any(size <= 0 for size in probabilities.shape):
        raise ValueError("probabilities must not be empty.")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two phases.")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point.")
    require_finite("probabilities", probabilities)
    if torch.any(probabilities < 0.0) or torch.any(probabilities > 1.0):
        raise ValueError("probabilities must be between 0 and 1.")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("probabilities must sum to one across phases.")

    try:
        run_lengths = tuple(lengths)
    except TypeError as exc:
        raise ValueError("lengths must be a sequence of integers.") from exc
    if not run_lengths:
        raise ValueError("lengths must not be empty.")
    for run_length in run_lengths:
        require_int("run length", run_length)
        if run_length < 2:
            raise ValueError("run lengths must be at least 2.")
    if len(set(run_lengths)) != len(run_lengths):
        raise ValueError("run lengths must be unique.")
    if tuple(sorted(run_lengths)) != run_lengths:
        raise ValueError("run lengths must be sorted.")
    if run_lengths[-1] > min(probabilities.shape[2:]):
        raise ValueError("run lengths must not exceed any spatial dimension.")
    return run_lengths
