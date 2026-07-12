import torch

from src.modeling.phases.representation import phase_levels
from src.common.tensors.validation import require_finite


def calc_phase_probs(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    phase_dim: int = 0,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if phase_dim < 0 or phase_dim > values.ndim:
        raise ValueError("phase_dim must be between 0 and values.ndim.")

    if not values.is_floating_point():
        raise ValueError("values must be a floating point tensor.")

    require_finite("values", values)

    levels = phase_levels(num_phases, device=values.device, dtype=values.dtype)
    level_shape = [1] * (values.ndim + 1)
    level_shape[phase_dim] = num_phases

    distance = values.unsqueeze(phase_dim) - levels.view(level_shape)

    return torch.softmax(-distance.pow(2) / temperature, dim=phase_dim)


def as_phase_probability_batch(
    probabilities: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    if probabilities.ndim == 3:
        probabilities = probabilities.unsqueeze(0)

    if probabilities.ndim != 4:
        raise ValueError(
            "phase probabilities must have shape [P, H, W] or [B, P, H, W]."
        )

    if probabilities.shape[1] != num_phases:
        raise ValueError("phase probability channels must match num_phases.")

    if any(size <= 0 for size in probabilities.shape):
        raise ValueError("phase probabilities must not be empty.")

    if not probabilities.is_floating_point():
        raise ValueError("phase probabilities must be floating point.")

    require_finite("phase probabilities", probabilities)

    if torch.any(probabilities < 0.0) or torch.any(probabilities > 1.0):
        raise ValueError("phase probabilities must be between 0 and 1.")

    sums = probabilities.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError("phase probabilities must sum to one across phases.")

    return probabilities


def sharpen_phase_probabilities(
    probabilities: torch.Tensor,
    *,
    num_phases: int,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    probabilities = as_phase_probability_batch(
        probabilities,
        num_phases=num_phases,
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    return torch.softmax(
        probabilities.clamp_min(tiny).log() / temperature,
        dim=1,
    )
