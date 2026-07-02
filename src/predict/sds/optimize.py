from collections.abc import Mapping, Sequence

import torch

from src.models import DDPM
from src.predict.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
    select_slice_batch,
)
from src.predict.sds.anchor import anchor_loss
from src.predict.sds.common import prepare_anchor_targets, prepare_inference_module
from src.predict.sds.core import sds_loss
from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.objective import descriptor_loss, descriptor_loss_per_sample
from src.predict.types import AnchorSlice
from src.predict.validation import validate_finite_tensor


def optimize_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPM,
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
    ddpm: DDPM,
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
        decoded = _decode_latent(vae, latent)

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
        decoded = _decode_latent(vae, latent).clamp(-1.0, 1.0)
        replace_slice(updated, axis, index, decoded)

    return updated, stats


def _optimize_slice_batch(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPM,
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
        decoded = _decode_latent_batch(vae, latent)

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
        decoded = _decode_latent_batch(vae, latent).clamp(-1.0, 1.0)
        replace_slice_batch(updated, axis, indices, decoded)

    return updated, stats


def _objective(
    latent: torch.Tensor,
    decoded: torch.Tensor,
    diffusion_model: torch.nn.Module,
    ddpm: DDPM,
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
    ddpm: DDPM,
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


def _validate_inputs(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    axis: int,
    index: int,
    steps: int,
    lr: float,
    sds_weight: float,
    anchor_weight: float,
    anchor_target: torch.Tensor | None,
    vf_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    tpc_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    sa_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if index < 0 or index >= volume.shape[axis]:
        raise ValueError("index must be inside the selected axis.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    _validate_optimization_contract(
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

    image_size = int(vae.image_size)
    if extract_slice(volume, axis, index).shape != torch.Size(
        [image_size, image_size]
    ):
        raise ValueError("selected slice shape must match vae.image_size.")


def _validate_optimization_contract(
    *,
    lr: float,
    sds_weight: float,
    anchor_weight: float,
    anchor_target: torch.Tensor | None,
    require_anchor_target: bool = True,
    vf_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    tpc_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    sa_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
) -> None:
    if lr <= 0.0:
        raise ValueError("lr must be positive.")

    for name, weight in (
        ("sds_weight", sds_weight),
        ("anchor_weight", anchor_weight),
        ("vf_weight", vf_weight),
        ("tpc_weight", tpc_weight),
        ("sa_weight", sa_weight),
        ("diffusivity_weight", diffusivity_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    if require_anchor_target and anchor_weight > 0.0 and anchor_target is None:
        raise ValueError("anchor_target is required when anchor_weight is positive.")

    if vf_weight > 0.0 and vf_targets is None:
        raise ValueError("vf_targets is required when vf_weight is positive.")

    if tpc_weight > 0.0 and tpc_targets is None:
        raise ValueError("tpc_targets is required when tpc_weight is positive.")

    if sa_weight > 0.0 and sa_targets is None:
        raise ValueError("sa_targets is required when sa_weight is positive.")

    if diffusivity_weight > 0.0 and diffusivity_targets is None:
        raise ValueError(
            "diffusivity_targets is required when diffusivity_weight is positive."
        )

    if diffusivity_weight > 0.0:
        if diffusivity_solver is None:
            raise ValueError("diffusivity_solver is required for diffusivity loss.")


def _validate_volume_inputs(
    volume: torch.Tensor,
    *,
    steps: int,
    slice_steps: int,
    sds_batch_size: int,
    slice_schedule: Sequence[tuple[int, int]] | None,
    anchors: Sequence[AnchorSlice] | None,
    anchor_weight: float,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    if any(size <= 0 for size in volume.shape):
        raise ValueError("volume dimensions must be positive.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    if slice_steps < 0:
        raise ValueError("slice_steps must be non-negative.")

    if not isinstance(sds_batch_size, int) or isinstance(sds_batch_size, bool):
        raise ValueError("sds_batch_size must be an integer.")

    if sds_batch_size <= 0:
        raise ValueError("sds_batch_size must be positive.")

    if slice_schedule is not None and len(slice_schedule) < steps * sds_batch_size:
        raise ValueError("slice_schedule must contain one entry per batched slice.")

    if anchor_weight > 0.0 and not anchors:
        raise ValueError("anchors are required when anchor_weight is positive.")

