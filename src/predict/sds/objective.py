from collections.abc import Mapping

import torch

from src.predict.sds.diffusivity import DiffusivitySolver, diffusivity_loss
from src.predict.sds.sa import surface_area_loss
from src.predict.sds.tpc import tpc_loss
from src.predict.sds.vf import volume_fraction_loss


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
    total = decoded.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

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
