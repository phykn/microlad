from collections.abc import Mapping, Sequence

import torch

from src.modeling.diffusion import DDPMProcess
from src.pipelines.guidance.conditioning.loss import anchor_loss
from src.pipelines.guidance.metrics.loss import descriptor_loss, sample_descriptor_loss
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.sds.prior import sds_loss


def slice_loss(
    latent: torch.Tensor,
    decoded: torch.Tensor,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_target: torch.Tensor | None,
    anchor_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    vf_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: ConductanceSolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    phase_probabilities: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = latent.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    if sds_weight > 0.0:
        loss, _ = sds_loss(latent, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
        total = total + sds_weight * loss
        stats["sds"] = (sds_weight * loss).detach()

    if anchor_weight > 0.0 and anchor_target is not None:
        anchor_kwargs = dict(
            mask=anchor_mask,
            num_phases=num_phases,
            temperature=temperature,
            weight=anchor_weight,
            phase_probabilities=phase_probabilities,
        )
        loss, _ = anchor_loss(decoded, anchor_target, **anchor_kwargs)
        total = total + loss
        stats["anchor"] = loss.detach()

    target_total, target_stats = descriptor_loss(
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
        phase_probabilities=phase_probabilities,
    )
    total = total + target_total
    stats.update(target_stats)

    stats["loss"] = total.detach()
    return total, stats


def batch_loss(
    latent: torch.Tensor,
    decoded: torch.Tensor,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_targets: Sequence[torch.Tensor | None],
    anchor_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    vf_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: ConductanceSolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    phase_probabilities: torch.Tensor | None = None,
    anchor_masks: Sequence[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = latent.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    if sds_weight > 0.0:
        loss, _ = sds_loss(latent, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
        total = total + sds_weight * loss
        stats["sds"] = (sds_weight * loss).detach()

    anchor_losses = []
    if anchor_weight > 0.0:
        masks = anchor_masks or [None] * len(anchor_targets)
        for batch_index, (decoded_slice, target, mask) in enumerate(
            zip(decoded, anchor_targets, masks, strict=True)
        ):
            if target is None:
                continue

            probability = (
                None
                if phase_probabilities is None
                else phase_probabilities[batch_index]
            )
            if mask is None:
                loss, _ = anchor_loss(
                    decoded_slice,
                    target,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                    phase_probabilities=probability,
                )
            else:
                loss, _ = anchor_loss(
                    decoded_slice,
                    target,
                    mask=mask,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                    phase_probabilities=probability,
                )
            anchor_losses.append(loss)

    if anchor_losses:
        loss = torch.stack(anchor_losses).mean()
        total = total + loss
        stats["anchor"] = loss.detach()

    target_total, target_stats = sample_descriptor_loss(
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
        phase_probabilities=phase_probabilities,
    )
    total = total + target_total
    stats.update(target_stats)

    stats["loss"] = total.detach()

    return total, stats
