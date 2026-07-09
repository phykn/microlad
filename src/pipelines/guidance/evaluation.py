from collections.abc import Mapping, Sequence

import torch

from src.modeling.diffusion import DDPMProcess
from src.pipelines.guidance.anchor_objective import anchor_loss
from src.pipelines.guidance.objective import descriptor_loss, descriptor_loss_per_sample
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.pipelines.guidance.prior import sds_loss
from src.common.tensors.validation import validate_finite_tensor

def _objective(
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
    diffusivity_solver: DiffusivitySolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = latent.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    if sds_weight > 0.0:
        loss, _ = sds_loss(latent, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
        total = total + sds_weight * loss
        stats["sds"] = (sds_weight * loss).detach()

    if anchor_weight > 0.0 and anchor_target is not None:
        loss, _ = anchor_loss(
            decoded,
            anchor_target,
            num_phases=num_phases,
            temperature=temperature,
            weight=anchor_weight,
        )
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
    )
    total = total + target_total
    stats.update(target_stats)

    stats["loss"] = total.detach()
    return total, stats


def _objective_batch(
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
    diffusivity_solver: DiffusivitySolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = latent.sum() * 0.0
    stats: dict[str, torch.Tensor] = {}

    if sds_weight > 0.0:
        loss, _ = sds_loss(latent, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
        total = total + sds_weight * loss
        stats["sds"] = (sds_weight * loss).detach()

    anchor_losses = []
    has_anchor_target = False
    if anchor_weight > 0.0:
        for decoded_slice, target in zip(decoded, anchor_targets, strict=True):
            if target is None:
                anchor_losses.append(decoded_slice.sum() * 0.0)
                continue

            loss, _ = anchor_loss(
                decoded_slice,
                target,
                num_phases=num_phases,
                temperature=temperature,
                weight=anchor_weight,
            )
            anchor_losses.append(loss)
            has_anchor_target = True

    if has_anchor_target:
        loss = torch.stack(anchor_losses).mean()
        total = total + loss
        stats["anchor"] = loss.detach()

    target_total, target_stats = descriptor_loss_per_sample(
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
    total = total + target_total
    stats.update(target_stats)

    stats["loss"] = total.detach()

    return total, stats


def _decode_latent(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latent)

    if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
        raise ValueError("vae.decode must return shape [1, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    validate_finite_tensor("decoded", decoded)

    return decoded[0, 0]


def _decode_latent_batch(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latent)

    if decoded.ndim != 4 or decoded.shape[0] != latent.shape[0] or decoded.shape[1] != 1:
        raise ValueError("vae.decode must return shape [B, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    validate_finite_tensor("decoded", decoded)

    return decoded[:, 0]
