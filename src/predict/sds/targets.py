from collections.abc import Mapping
from numbers import Integral

import torch

from src.tensors.validation import validate_finite_tensor, validate_floating_dtype


def phase_vector_target(
    targets: Mapping[int, float] | torch.Tensor,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    require_sum_one: bool = False,
) -> torch.Tensor:
    validate_floating_dtype("dtype", dtype)

    if isinstance(targets, torch.Tensor):
        target = targets.to(device=device, dtype=dtype)
    else:
        expected_keys = set(range(num_phases))
        seen_keys: set[int] = set()
        target = torch.zeros(num_phases, device=device, dtype=dtype)

        for phase, value in targets.items():
            if not isinstance(phase, Integral) or isinstance(phase, bool):
                raise ValueError("targets phase indices must be integers.")

            phase = int(phase)
            if phase < 0 or phase >= num_phases:
                raise ValueError("targets must contain phase indices within num_phases.")

            seen_keys.add(phase)
            target[phase] = float(value)

        if seen_keys != expected_keys:
            raise ValueError(f"targets must contain one {label} per phase.")

    if target.shape != torch.Size([num_phases]):
        raise ValueError(f"targets must have one {label} per phase.")

    validate_finite_tensor("targets", target)

    if torch.any(target < 0):
        raise ValueError("targets must be non-negative.")

    if require_sum_one and not torch.allclose(
        target.sum(),
        target.new_tensor(1.0),
        atol=1e-6,
    ):
        raise ValueError("targets must sum to 1.")

    return target
