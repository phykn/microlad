from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch

from src.modeling.phases.quantization import MAX_UINT8_PHASES
from src.pipelines.guidance.conditioning.images import prepare_phase_image
from src.pipelines.guidance.metrics.conductance import (
    ConductanceSolver,
    compute_conductance,
)
from src.pipelines.guidance.metrics.correlation import compute_correlation
from src.pipelines.guidance.metrics.fraction import compute_phase_fraction
from src.pipelines.guidance.metrics.interface import compute_interface_density
from src.validation import require_finite_number, require_int


class DescriptorTargets(TypedDict, total=False):
    fraction_targets: torch.Tensor
    tpc_targets: torch.Tensor
    sa_targets: torch.Tensor
    diffusivity_targets: torch.Tensor
    diffusivity_solver: ConductanceSolver


def build_descriptor_targets(
    labels: torch.Tensor | None,
    *,
    num_phases: int,
    use_fraction: bool = False,
    use_tpc: bool = False,
    use_sa: bool = False,
    use_diffusivity: bool = False,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    diffusivity_grid_size: int | tuple[int, int] | None = None,
    low_phase_conductivity: float = 0.0,
) -> DescriptorTargets:
    _validate_options(
        num_phases=num_phases,
        temperature=temperature,
        sa_kernel_size=sa_kernel_size,
        sa_sigma=sa_sigma,
        use_diffusivity=use_diffusivity,
        diffusivity_grid_size=diffusivity_grid_size,
        low_phase_conductivity=low_phase_conductivity,
    )
    if not (use_fraction or use_tpc or use_sa or use_diffusivity):
        return {}

    values = _validate_target_labels(labels, num_phases=num_phases).float()
    targets: DescriptorTargets = {}

    if use_fraction:
        targets["fraction_targets"] = compute_phase_fraction(
            values,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()

    if use_tpc:
        targets["tpc_targets"] = compute_correlation(
            values,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()

    if use_sa:
        targets["sa_targets"] = compute_interface_density(
            values,
            num_phases=num_phases,
            temperature=temperature,
            kernel_size=sa_kernel_size,
            sigma=sa_sigma,
        ).detach()

    if use_diffusivity:
        height, width = _diffusivity_shape(diffusivity_grid_size)
        solver = ConductanceSolver(
            height=height,
            width=width,
            low_cond=low_phase_conductivity,
        )
        targets["diffusivity_targets"] = compute_conductance(
            values,
            solver=solver,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()
        targets["diffusivity_solver"] = solver

    return targets


def prepare_target_images(
    images: Sequence[np.ndarray],
    *,
    num_phases: int,
    segment: bool = False,
) -> torch.Tensor:
    if not images:
        raise ValueError("images are required when target options are enabled.")

    phases = []
    shape = None
    for image in images:
        phase = prepare_phase_image(
            image,
            num_phases=num_phases,
            segment=segment,
            name="target image",
        )
        if shape is None:
            shape = phase.shape
        elif phase.shape != shape:
            raise ValueError("target images must have the same shape.")

        phases.append(phase)

    values = np.stack(phases).astype(np.int64)
    return torch.from_numpy(values.copy()).long()


def _validate_target_labels(
    labels: torch.Tensor | None,
    *,
    num_phases: int,
) -> torch.Tensor:
    if labels is None:
        raise ValueError("target labels are required when target options are enabled.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("target labels must be a tensor.")
    if labels.ndim != 3 or labels.shape[0] == 0 or min(labels.shape[1:]) <= 0:
        raise ValueError("target labels must have shape [B, H, W].")
    if labels.dtype != torch.long:
        raise ValueError("target labels must use torch.long dtype.")
    if labels.min().item() < 0 or labels.max().item() >= num_phases:
        raise ValueError(
            f"target labels must contain values from 0 to {num_phases - 1}."
        )
    return labels


def _validate_options(
    *,
    num_phases: int,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    use_diffusivity: bool,
    diffusivity_grid_size: int | tuple[int, int] | None,
    low_phase_conductivity: float,
) -> None:
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 images."
        )

    _require_positive("temperature", temperature)

    require_int("sa_kernel_size", sa_kernel_size)
    if sa_kernel_size <= 0 or sa_kernel_size % 2 == 0:
        raise ValueError("sa_kernel_size must be a positive odd integer.")

    _require_positive("sa_sigma", sa_sigma)

    if use_diffusivity:
        _diffusivity_shape(diffusivity_grid_size)

    require_finite_number("low_phase_conductivity", low_phase_conductivity)
    if low_phase_conductivity < 0.0 or low_phase_conductivity > 1.0:
        raise ValueError("low_phase_conductivity must be between 0 and 1.")


def _diffusivity_shape(size: int | tuple[int, int] | None) -> tuple[int, int]:
    if size is None:
        raise ValueError(
            "diffusivity_grid_size is required when use_diffusivity is True."
        )

    if isinstance(size, int) and not isinstance(size, bool):
        height = width = size
    elif isinstance(size, tuple) and len(size) == 2:
        height, width = size
    else:
        raise ValueError("diffusivity_grid_size must be an integer or (height, width).")

    require_int("diffusivity_grid_size height", height)
    require_int("diffusivity_grid_size width", width)

    if height < 2:
        raise ValueError("diffusivity_grid_size height must be at least 2.")

    if width < 2:
        raise ValueError("diffusivity_grid_size width must be at least 2.")

    return int(height), int(width)


def _require_positive(name: str, value: float) -> None:
    require_finite_number(name, value)

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
