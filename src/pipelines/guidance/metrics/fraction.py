from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.modeling.phases.relaxation import (
    calc_phase_probs,
    sharpen_phase_probabilities,
)
from src.pipelines.guidance.metrics.targets import build_phase_target
from src.validation import require_finite, require_finite_number, require_int


def phase_fraction_loss(
    values: torch.Tensor,
    targets: Mapping[int, float] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
    phase_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.numel() == 0:
        raise ValueError("values must be non-empty.")

    require_finite("values", values)
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    require_finite_number("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    require_finite_number("weight", weight)
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_fraction = compute_phase_fraction(
        values,
        num_phases=num_phases,
        temperature=temperature,
        phase_probabilities=phase_probabilities,
    )
    target = build_phase_target(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
        label="fraction",
        require_sum_one=True,
    )

    loss = weight * F.mse_loss(actual_fraction, target)

    stats = {
        "actual_fraction": actual_fraction.detach(),
        "target_fraction": target.detach(),
    }

    return loss, stats


def compute_phase_fraction(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    phase_probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("values must be non-empty.")

    require_finite("values", values)
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    require_finite_number("temperature", temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if phase_probabilities is not None:
        probability = sharpen_phase_probabilities(
            phase_probabilities,
            num_phases=num_phases,
            temperature=temperature,
        )
        return probability.mean(dim=(0, 2, 3))

    probability = calc_phase_probs(
        values.reshape(-1),
        num_phases=num_phases,
        temperature=temperature,
        phase_dim=0,
    )

    return probability.mean(dim=1)
