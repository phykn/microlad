from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.predict.sds.phase import soft_phase_probability
from src.predict.sds.targets import phase_vector_target


def volume_fraction_loss(
    values: torch.Tensor,
    targets: Mapping[int, float] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_vf = compute_volume_fraction(
        values,
        num_phases=num_phases,
        temperature=temperature,
    )
    target = phase_vector_target(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
        label="fraction",
        require_sum_one=True,
    )
    loss = weight * F.mse_loss(actual_vf, target)

    stats = {
        "actual_vf": actual_vf.detach(),
        "target_vf": target.detach(),
    }
    return loss, stats


def compute_volume_fraction(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    probability = soft_phase_probability(
        values.reshape(-1),
        num_phases=num_phases,
        temperature=temperature,
        phase_dim=0,
    )
    return probability.mean(dim=1)
