from collections.abc import Mapping

import torch

from src.pipeline.predict.guidance.metrics.conductance import (
    ConductanceSolver,
    conductance_loss,
)
from src.pipeline.predict.guidance.metrics.correlation import correlation_loss
from src.pipeline.predict.guidance.metrics.fraction import phase_fraction_loss
from src.pipeline.predict.guidance.metrics.interface import interface_loss
from src.validation import require_finite


def _validate_descriptor(name: str, weight: float, targets) -> None:
    if weight < 0.0:
        raise ValueError(f"{name}_weight must be non-negative.")

    if weight > 0.0 and targets is None:
        raise ValueError(f"{name}_targets are required when {name}_weight is positive.")


def descriptor_loss(
    decoded: torch.Tensor,
    *,
    num_phases: int,
    fraction_targets: Mapping[int, float] | torch.Tensor | None = None,
    fraction_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    phase_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    require_finite("decoded", decoded)

    total = decoded.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    _validate_descriptor("fraction", fraction_weight, fraction_targets)
    _validate_descriptor("tpc", tpc_weight, tpc_targets)
    _validate_descriptor("sa", sa_weight, sa_targets)
    _validate_descriptor(
        "diffusivity",
        diffusivity_weight,
        diffusivity_targets,
    )

    if fraction_weight > 0.0 and fraction_targets is not None:
        loss, _ = phase_fraction_loss(
            decoded,
            fraction_targets,
            num_phases=num_phases,
            temperature=temperature,
            weight=fraction_weight,
            phase_probabilities=phase_probabilities,
        )
        total = total + loss
        stats["fraction"] = loss.detach()

    if tpc_weight > 0.0 and tpc_targets is not None:
        loss, _ = correlation_loss(
            decoded,
            tpc_targets,
            num_phases=num_phases,
            temperature=temperature,
            weight=tpc_weight,
            phase_probabilities=phase_probabilities,
        )
        total = total + loss
        stats["tpc"] = loss.detach()

    if sa_weight > 0.0 and sa_targets is not None:
        loss, _ = interface_loss(
            decoded,
            sa_targets,
            num_phases=num_phases,
            temperature=temperature,
            kernel_size=sa_kernel_size,
            sigma=sa_sigma,
            weight=sa_weight,
            phase_probabilities=phase_probabilities,
        )
        total = total + loss
        stats["sa"] = loss.detach()

    if diffusivity_weight > 0.0 and diffusivity_targets is not None:
        if diffusivity_solver is None:
            raise ValueError("diffusivity_solver is required for diffusivity loss.")

        loss, _ = conductance_loss(
            decoded,
            diffusivity_targets,
            solver=diffusivity_solver,
            num_phases=num_phases,
            temperature=temperature,
            weight=diffusivity_weight,
            phase_probabilities=phase_probabilities,
        )
        total = total + loss
        stats["diffusivity"] = loss.detach()

    return total, stats


def sample_descriptor_loss(
    decoded: torch.Tensor,
    *,
    num_phases: int,
    fraction_targets: Mapping[int, float] | torch.Tensor | None = None,
    fraction_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    phase_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return descriptor_loss(
        decoded,
        num_phases=num_phases,
        fraction_targets=fraction_targets,
        fraction_weight=fraction_weight,
        tpc_targets=tpc_targets,
        tpc_weight=tpc_weight,
        sa_targets=sa_targets,
        sa_weight=sa_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
        diffusivity_weight=diffusivity_weight,
        temperature=temperature,
        sa_kernel_size=sa_kernel_size,
        sa_sigma=sa_sigma,
        phase_probabilities=phase_probabilities,
    )
