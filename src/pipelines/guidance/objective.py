from collections.abc import Mapping

import torch

from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver, diffusivity_loss
from src.pipelines.guidance.descriptors.surface_area import surface_area_loss
from src.pipelines.guidance.descriptors.two_point_correlation import tpc_loss
from src.pipelines.guidance.descriptors.volume_fraction import volume_fraction_loss
from src.common.tensors.validation import require_finite


def _validate_descriptor(name: str, weight: float, targets) -> None:
    if weight < 0.0:
        raise ValueError(f"{name}_weight must be non-negative.")

    if weight > 0.0 and targets is None:
        raise ValueError(f"{name}_targets are required when {name}_weight is positive.")


def descriptor_loss(
    decoded: torch.Tensor,
    *,
    num_phases: int,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    require_finite("decoded", decoded)

    total = decoded.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    _validate_descriptor("vf", vf_weight, vf_targets)
    _validate_descriptor("tpc", tpc_weight, tpc_targets)
    _validate_descriptor("sa", sa_weight, sa_targets)
    _validate_descriptor(
        "diffusivity",
        diffusivity_weight,
        diffusivity_targets,
    )

    if vf_weight > 0.0 and vf_targets is not None:
        loss, _ = volume_fraction_loss(
            decoded,
            vf_targets,
            num_phases=num_phases,
            temperature=temperature,
            weight=vf_weight,
        )
        total = total + loss
        stats["vf"] = loss.detach()

    if tpc_weight > 0.0 and tpc_targets is not None:
        loss, _ = tpc_loss(
            decoded,
            tpc_targets,
            num_phases=num_phases,
            temperature=temperature,
            weight=tpc_weight,
        )
        total = total + loss
        stats["tpc"] = loss.detach()

    if sa_weight > 0.0 and sa_targets is not None:
        loss, _ = surface_area_loss(
            decoded,
            sa_targets,
            num_phases=num_phases,
            temperature=temperature,
            kernel_size=sa_kernel_size,
            sigma=sa_sigma,
            weight=sa_weight,
        )
        total = total + loss
        stats["sa"] = loss.detach()

    if diffusivity_weight > 0.0 and diffusivity_targets is not None:
        if diffusivity_solver is None:
            raise ValueError("diffusivity_solver is required for diffusivity loss.")

        loss, _ = diffusivity_loss(
            decoded,
            diffusivity_targets,
            solver=diffusivity_solver,
            num_phases=num_phases,
            temperature=temperature,
            weight=diffusivity_weight,
        )
        total = total + loss
        stats["diffusivity"] = loss.detach()

    return total, stats


def sample_descriptor_loss(
    decoded: torch.Tensor,
    *,
    num_phases: int,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if decoded.ndim < 3:
        return descriptor_loss(
            decoded,
            num_phases=num_phases,
            vf_targets=vf_targets,
            vf_weight=vf_weight,
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
        )

    losses = []
    history: dict[str, list[torch.Tensor]] = {}

    for sample in decoded:
        loss, stats = descriptor_loss(
            sample,
            num_phases=num_phases,
            vf_targets=vf_targets,
            vf_weight=vf_weight,
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
        )
        losses.append(loss)

        for key, value in stats.items():
            history.setdefault(key, []).append(value)

    total = torch.stack(losses).mean()
    mean_stats = {
        key: torch.stack(values).mean(dim=0)
        for key, values in history.items()
        if values
    }

    return total, mean_stats
