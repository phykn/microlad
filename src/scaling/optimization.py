import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.diffusion import DDPMProcess
from src.scaling.blending import blend_window
from src.reconstruction.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
    select_slice_batch,
)
from src.guidance.anchor_objective import anchor_loss, masked_anchor_loss
from src.guidance.preparation import prepare_anchor_targets, prepare_inference_module
from src.guidance.prior import sds_loss
from src.guidance.physics.diffusivity import DiffusivitySolver
from src.guidance.objective import descriptor_loss, descriptor_loss_per_sample
from src.guidance.conditioning.model import AnchorSlice
from src.scaling.tiles import normalized_tile_weights, tile_grid
from src.scaling.local_objective import (
    _decode_tiled_image,
    _decode_tiled_image_batch,
    _local_prior_objective,
    _local_prior_objective_batch,
    _mean_stats,
    _record_stats,
)
from src.scaling.validation import _as_anchor_image, _tensor_map, _validate_inputs

from src.tensors.validation import validate_finite_tensor, validate_floating_dtype


def optimize_large_volume(
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
    anchor_targets: Mapping[tuple[int, int], torch.Tensor] | None = None,
    anchor_masks: Mapping[tuple[int, int], torch.Tensor] | None = None,
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
    descriptor_tile_size: int | None = None,
    temperature: float = 0.1,
    tile_overlap: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_inputs(
        volume,
        steps=steps,
        slice_steps=slice_steps,
        sds_batch_size=sds_batch_size,
        lr=lr,
        slice_schedule=slice_schedule,
        anchors=anchors,
        anchor_targets=anchor_targets,
        anchor_masks=anchor_masks,
        anchor_weight=anchor_weight,
        sds_weight=sds_weight,
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
        num_phases=num_phases,
    )
    prepared_targets = prepare_anchor_targets(
        vae,
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=anchor_segment,
        device=volume.device,
        dtype=volume.dtype,
        tile_overlap=tile_overlap,
    )
    prepared_targets.update(
        _tensor_map(
            anchor_targets,
            device=volume.device,
            dtype=volume.dtype,
        )
    )
    prepared_masks = _tensor_map(
        anchor_masks,
        device=volume.device,
        dtype=volume.dtype,
    )

    prepare_inference_module(vae)
    prepare_inference_module(diffusion_model)

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
            target = prepared_targets.get((axis, index))
            mask = prepared_masks.get((axis, index))
            weight = anchor_weight if target is not None else 0.0
            image = extract_slice(updated, axis, index)

            refined, stats = _optimize_large_slice(
                image,
                vae,
                diffusion_model,
                ddpm,
                steps=slice_steps,
                lr=lr,
                t_min=t_min,
                t_max=t_max,
                num_phases=num_phases,
                sds_weight=sds_weight,
                anchor_target=target,
                anchor_mask=mask,
                anchor_weight=weight,
                vf_targets=vf_targets,
                vf_weight=vf_weight,
                tpc_targets=tpc_targets,
                tpc_weight=tpc_weight,
                sa_targets=sa_targets,
                sa_weight=sa_weight,
                diffusivity_targets=diffusivity_targets,
                diffusivity_solver=diffusivity_solver,
                diffusivity_weight=diffusivity_weight,
                descriptor_tile_size=descriptor_tile_size,
                temperature=temperature,
                tile_overlap=tile_overlap,
            )
            replace_slice(updated, axis, index, refined)
        else:
            images = extract_slice_batch(updated, axis, indices)
            targets = [prepared_targets.get((axis, index)) for index in indices]
            masks = [prepared_masks.get((axis, index)) for index in indices]

            refined, stats = _optimize_large_slice_batch(
                images,
                vae,
                diffusion_model,
                ddpm,
                steps=slice_steps,
                lr=lr,
                t_min=t_min,
                t_max=t_max,
                num_phases=num_phases,
                sds_weight=sds_weight,
                anchor_targets=targets,
                anchor_masks=masks,
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
                descriptor_tile_size=descriptor_tile_size,
                temperature=temperature,
                tile_overlap=tile_overlap,
            )
            replace_slice_batch(updated, axis, indices, refined)

        for key, value in stats.items():
            history.setdefault(key, []).append(value.detach())

    stats = {
        key: torch.stack(values).mean()
        for key, values in history.items()
        if values
    }
    stats["steps"] = torch.tensor(steps, device=updated.device)

    return updated, stats


def _optimize_large_slice(
    image: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_target: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
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
    descriptor_tile_size: int | None,
    temperature: float,
    tile_overlap: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_image = _as_anchor_image(anchor_target) if anchor_target is not None else None
    target_mask = _as_anchor_image(anchor_mask) if anchor_mask is not None else None

    if descriptor_tile_size is not None and descriptor_tile_size != int(vae.image_size):
        raise ValueError("descriptor_tile_size must match vae.image_size.")

    use_tile_descriptor = descriptor_tile_size is not None

    if steps == 0:
        return image.float(), {}

    image_param = image.detach().clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([image_param], lr=lr)
    history: dict[str, list[torch.Tensor]] = {}

    for _ in range(steps):
        optimizer.zero_grad()

        decoded, total, stats = _local_prior_objective(
            image_param,
            vae,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_target=target_image,
            anchor_mask=target_mask,
            anchor_weight=anchor_weight,
            temperature=temperature,
            tile_overlap=tile_overlap,
            vf_targets=vf_targets if use_tile_descriptor else None,
            vf_weight=vf_weight if use_tile_descriptor else 0.0,
            tpc_targets=tpc_targets if use_tile_descriptor else None,
            tpc_weight=tpc_weight if use_tile_descriptor else 0.0,
            sa_targets=sa_targets if use_tile_descriptor else None,
            sa_weight=sa_weight if use_tile_descriptor else 0.0,
            diffusivity_targets=diffusivity_targets if use_tile_descriptor else None,
            diffusivity_solver=diffusivity_solver if use_tile_descriptor else None,
            diffusivity_weight=diffusivity_weight if use_tile_descriptor else 0.0,
        )
        if not use_tile_descriptor:
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
            )
            total = total + target_total
            stats.update(target_stats)
        stats["loss"] = total.detach()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            image_param.clamp_(0.0, float(num_phases - 1))
        _record_stats(history, stats)

    with torch.no_grad():
        decoded = _decode_tiled_image(
            image_param.detach(),
            vae,
            tile_overlap=tile_overlap,
        )
    return decoded, _mean_stats(history)


def _optimize_large_slice_batch(
    images: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_targets: Sequence[torch.Tensor | None],
    anchor_masks: Sequence[torch.Tensor | None],
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
    descriptor_tile_size: int | None,
    temperature: float,
    tile_overlap: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if images.ndim != 3:
        raise ValueError("images must have shape [B, H, W].")

    if len(anchor_targets) != images.shape[0] or len(anchor_masks) != images.shape[0]:
        raise ValueError("anchor target batch size must match images.")

    if descriptor_tile_size is not None and descriptor_tile_size != int(vae.image_size):
        raise ValueError("descriptor_tile_size must match vae.image_size.")

    use_tile_descriptor = descriptor_tile_size is not None

    if steps == 0:
        return images.float(), {}

    image_param = images.detach().clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([image_param], lr=lr)
    history: dict[str, list[torch.Tensor]] = {}

    for _ in range(steps):
        optimizer.zero_grad()

        decoded, total, stats = _local_prior_objective_batch(
            image_param,
            vae,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_targets=anchor_targets,
            anchor_masks=anchor_masks,
            anchor_weight=anchor_weight,
            temperature=temperature,
            tile_overlap=tile_overlap,
            vf_targets=vf_targets if use_tile_descriptor else None,
            vf_weight=vf_weight if use_tile_descriptor else 0.0,
            tpc_targets=tpc_targets if use_tile_descriptor else None,
            tpc_weight=tpc_weight if use_tile_descriptor else 0.0,
            sa_targets=sa_targets if use_tile_descriptor else None,
            sa_weight=sa_weight if use_tile_descriptor else 0.0,
            diffusivity_targets=diffusivity_targets if use_tile_descriptor else None,
            diffusivity_solver=diffusivity_solver if use_tile_descriptor else None,
            diffusivity_weight=diffusivity_weight if use_tile_descriptor else 0.0,
        )

        if not use_tile_descriptor:
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
            )
            total = total + target_total
            stats.update(target_stats)

        stats["loss"] = total.detach()
        total.backward()
        optimizer.step()

        with torch.no_grad():
            image_param.clamp_(0.0, float(num_phases - 1))

        _record_stats(history, stats)

    with torch.no_grad():
        decoded = _decode_tiled_image_batch(
            image_param.detach(),
            vae,
            tile_overlap=tile_overlap,
        )

    return decoded, _mean_stats(history)
