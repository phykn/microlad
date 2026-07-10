from collections.abc import Mapping, Sequence

import torch

from src.modeling.diffusion import DDPMProcess
from src.pipelines.reconstruction.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
    select_slice_batch,
)
from src.pipelines.guidance.anchor_objective import anchor_loss
from src.pipelines.guidance.preparation import prepare_anchor_targets, prepare_inference_module
from src.pipelines.guidance.prior import sds_loss
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.pipelines.guidance.objective import descriptor_loss, descriptor_loss_per_sample
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.evaluation import (
    _objective,
    _objective_batch,
)
from src.pipelines.reconstruction.volume import decode_latent, decode_latents
from src.pipelines.guidance.validation import (
    _validate_inputs,
    _validate_optimization_contract,
    _validate_volume_inputs,
)

from src.common.tensors.validation import validate_finite_tensor


def optimize_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    slice_steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_batch_size: int = 1,
    slice_schedule: Sequence[tuple[int, int]] | None = None,
    anchors: Sequence[AnchorSlice] | None = None,
    anchor_segment: bool = False,
    sds_weight: float = 1.0,
    anchor_weight: float = 0.0,
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
    _validate_volume_inputs(
        volume,
        steps=steps,
        slice_steps=slice_steps,
        sds_batch_size=sds_batch_size,
        slice_schedule=slice_schedule,
        anchors=anchors,
        anchor_weight=anchor_weight,
    )
    _validate_optimization_contract(
        lr=lr,
        sds_weight=sds_weight,
        anchor_weight=anchor_weight,
        anchor_target=None,
        require_anchor_target=False,
        vf_weight=vf_weight,
        vf_targets=vf_targets,
        tpc_weight=tpc_weight,
        tpc_targets=tpc_targets,
        sa_weight=sa_weight,
        sa_targets=sa_targets,
        diffusivity_weight=diffusivity_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
    )
    anchor_targets = prepare_anchor_targets(
        vae,
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=anchor_segment,
        device=volume.device,
        dtype=volume.dtype,
    )

    updated = volume.clone().float()
    history: dict[str, list[torch.Tensor]] = {}

    for step in range(steps):
        axis, indices = select_slice_batch(
            updated,
            step,
            slice_schedule,
            sds_batch_size,
        )

        if len(indices) == 1:
            index = indices[0]
            anchor_target = anchor_targets.get((axis, index))
            current_anchor_weight = anchor_weight if anchor_target is not None else 0.0

            updated, step_stats = optimize_slice(
                updated,
                vae,
                diffusion_model,
                ddpm,
                axis=axis,
                index=index,
                steps=slice_steps,
                lr=lr,
                t_min=t_min,
                t_max=t_max,
                num_phases=num_phases,
                sds_weight=sds_weight,
                anchor_target=anchor_target,
                anchor_weight=current_anchor_weight,
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
        else:
            updated, step_stats = _optimize_slice_batch(
                updated,
                vae,
                diffusion_model,
                ddpm,
                axis=axis,
                indices=indices,
                steps=slice_steps,
                lr=lr,
                t_min=t_min,
                t_max=t_max,
                num_phases=num_phases,
                sds_weight=sds_weight,
                anchor_targets=[
                    anchor_targets.get((axis, index))
                    for index in indices
                ],
                anchor_weight=anchor_weight,
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

        for key, value in step_stats.items():
            history.setdefault(key, []).append(value.detach())

    stats = {
        key: torch.stack(values).mean()
        for key, values in history.items()
        if values
    }
    stats["steps"] = torch.tensor(steps, device=updated.device)

    return updated, stats


def optimize_slice(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    axis: int,
    index: int,
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float = 1.0,
    anchor_target: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
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
    _validate_inputs(
        volume,
        vae,
        axis=axis,
        index=index,
        steps=steps,
        lr=lr,
        sds_weight=sds_weight,
        anchor_weight=anchor_weight,
        anchor_target=anchor_target,
        vf_weight=vf_weight,
        vf_targets=vf_targets,
        tpc_weight=tpc_weight,
        tpc_targets=tpc_targets,
        sa_weight=sa_weight,
        sa_targets=sa_targets,
        diffusivity_weight=diffusivity_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
    )

    updated = volume.clone().float()
    if steps == 0:
        return updated, {}

    prepare_inference_module(vae)
    prepare_inference_module(diffusion_model)

    image = extract_slice(updated, axis, index).view(
        1,
        1,
        int(vae.image_size),
        int(vae.image_size),
    )
    mu, _ = vae.encode(image)

    if mu.ndim != 4:
        raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

    validate_finite_tensor("latent", mu)

    latent = mu.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    stats: dict[str, torch.Tensor] = {}

    for _ in range(steps):
        optimizer.zero_grad()
        decoded = decode_latent(vae, latent)

        total, stats = _objective(
            latent,
            decoded,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_target=anchor_target,
            anchor_weight=anchor_weight,
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
        total.backward()
        optimizer.step()

    with torch.no_grad():
        decoded = decode_latent(vae, latent)
        replace_slice(updated, axis, index, decoded)

    return updated, stats


def _optimize_slice_batch(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    axis: int,
    indices: Sequence[int],
    steps: int,
    lr: float,
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
    updated = volume.clone().float()
    if steps == 0:
        return updated, {}

    prepare_inference_module(vae)
    prepare_inference_module(diffusion_model)

    images = extract_slice_batch(updated, axis, indices)
    image_size = int(vae.image_size)
    if images.shape[-2:] != torch.Size([image_size, image_size]):
        raise ValueError("selected slice shape must match vae.image_size.")

    latent, _ = vae.encode(images.view(len(indices), 1, image_size, image_size))

    if latent.ndim != 4 or latent.shape[0] != len(indices):
        raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

    validate_finite_tensor("latent", latent)

    latent = latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    stats: dict[str, torch.Tensor] = {}

    for _ in range(steps):
        optimizer.zero_grad()
        decoded = decode_latents(vae, latent)

        total, stats = _objective_batch(
            latent,
            decoded,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_targets=anchor_targets,
            anchor_weight=anchor_weight,
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
        total.backward()
        optimizer.step()

    with torch.no_grad():
        decoded = decode_latents(vae, latent)
        replace_slice_batch(updated, axis, indices, decoded)

    return updated, stats
