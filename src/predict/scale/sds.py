import math
from collections.abc import Mapping, Sequence

import torch

from src.models import DDPM
from src.predict.blend import blend_window
from src.predict.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
    select_slice_batch,
)
from src.predict.sds.anchor import anchor_loss, masked_anchor_loss
from src.predict.sds.common import prepare_anchor_targets, prepare_inference_module
from src.predict.sds.core import sds_loss
from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.objective import descriptor_loss, descriptor_loss_per_sample
from src.predict.types import AnchorSlice
from src.predict.scale.tiles import tile_grid
from src.predict.validation import validate_finite_tensor, validate_floating_dtype


def optimize_large_volume(
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
    ddpm: DDPM,
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
    ddpm: DDPM,
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


def _local_prior_objective(
    image: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPM,
    *,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_target: torch.Tensor | None,
    anchor_weight: float,
    temperature: float,
    tile_overlap: int,
    anchor_mask: torch.Tensor | None = None,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tile_size = int(vae.image_size)
    height, width = int(image.shape[0]), int(image.shape[1])
    out = image.new_zeros(image.shape)
    weight_sum = image.detach().new_zeros(image.shape)
    window = _tile_blend_window(
        tile_size,
        tile_overlap,
        reference=image,
    )
    total = image.sum() * 0.0
    tile_count = 0
    history: dict[str, list[torch.Tensor]] = {}

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        tile_count += 1
        patch = image[row : row + tile_size, col : col + tile_size].reshape(
            1,
            1,
            tile_size,
            tile_size,
        )
        mu, _ = vae.encode(patch)

        if mu.ndim != 4:
            raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

        validate_finite_tensor("latent", mu)

        decoded = _decode_latent(vae, mu)
        out[row : row + tile_size, col : col + tile_size] = (
            out[row : row + tile_size, col : col + tile_size] + decoded * window
        )
        weight_sum[row : row + tile_size, col : col + tile_size] = (
            weight_sum[row : row + tile_size, col : col + tile_size] + window
        )

        stats: dict[str, torch.Tensor] = {}
        if sds_weight > 0.0:
            loss, _ = sds_loss(mu, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
            weighted = sds_weight * loss
            total = total + weighted
            stats["sds"] = weighted.detach()

        if anchor_weight > 0.0 and anchor_target is not None:
            target_patch = anchor_target[row : row + tile_size, col : col + tile_size]
            mask_patch = (
                None
                if anchor_mask is None
                else anchor_mask[row : row + tile_size, col : col + tile_size]
            )
            has_anchor_pixels = (
                mask_patch is None or bool((mask_patch > 0).any().item())
            )

            if has_anchor_pixels:
                if mask_patch is None:
                    loss, _ = anchor_loss(
                        decoded,
                        target_patch,
                        num_phases=num_phases,
                        temperature=temperature,
                        weight=anchor_weight,
                    )
                else:
                    loss, _ = masked_anchor_loss(
                        decoded,
                        target_patch,
                        mask_patch,
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
        )
        total = total + target_total
        stats.update(target_stats)
        _record_stats(history, stats)

    return (
        _weighted_average(out, weight_sum),
        total / max(tile_count, 1),
        _mean_stats(history),
    )


def _local_prior_objective_batch(
    images: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPM,
    *,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_targets: Sequence[torch.Tensor | None],
    anchor_weight: float,
    temperature: float,
    tile_overlap: int,
    anchor_masks: Sequence[torch.Tensor | None],
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    tile_size = int(vae.image_size)
    out = images.new_zeros(images.shape)
    weight_sum = images.detach().new_zeros(images.shape)
    window = _tile_blend_window(
        tile_size,
        tile_overlap,
        reference=images,
    )
    total = images.sum() * 0.0
    tile_count = 0
    history: dict[str, list[torch.Tensor]] = {}

    target_images = [
        _as_anchor_image(target) if target is not None else None
        for target in anchor_targets
    ]
    mask_images = [
        _as_anchor_image(mask) if mask is not None else None
        for mask in anchor_masks
    ]

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        tile_count += 1
        patches = images[:, row : row + tile_size, col : col + tile_size].reshape(
            batch_size,
            1,
            tile_size,
            tile_size,
        )
        mu, _ = vae.encode(patches)

        if mu.ndim != 4 or mu.shape[0] != batch_size:
            raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

        validate_finite_tensor("latent", mu)

        decoded = _decode_latent_batch(vae, mu)
        out[:, row : row + tile_size, col : col + tile_size] = (
            out[:, row : row + tile_size, col : col + tile_size] + decoded * window
        )
        weight_sum[:, row : row + tile_size, col : col + tile_size] = (
            weight_sum[:, row : row + tile_size, col : col + tile_size] + window
        )

        stats: dict[str, torch.Tensor] = {}
        if sds_weight > 0.0:
            loss, _ = sds_loss(mu, diffusion_model, ddpm, t_min=t_min, t_max=t_max)
            weighted = sds_weight * loss
            total = total + weighted
            stats["sds"] = weighted.detach()

        anchor_losses = []
        has_active_anchor_pixels = False
        if anchor_weight > 0.0:
            for slice_index, decoded_slice in enumerate(decoded):
                target_image = target_images[slice_index]
                if target_image is None:
                    anchor_losses.append(decoded_slice.sum() * 0.0)
                    continue

                target_patch = target_image[
                    row : row + tile_size,
                    col : col + tile_size,
                ]
                mask_image = mask_images[slice_index]
                mask_patch = (
                    None
                    if mask_image is None
                    else mask_image[row : row + tile_size, col : col + tile_size]
                )
                slice_has_anchor_pixels = (
                    mask_patch is None or bool((mask_patch > 0).any().item())
                )

                if not slice_has_anchor_pixels:
                    anchor_losses.append(decoded_slice.sum() * 0.0)
                    continue

                if mask_patch is None:
                    loss, _ = anchor_loss(
                        decoded_slice,
                        target_patch,
                        num_phases=num_phases,
                        temperature=temperature,
                        weight=anchor_weight,
                    )
                else:
                    loss, _ = masked_anchor_loss(
                        decoded_slice,
                        target_patch,
                        mask_patch,
                        num_phases=num_phases,
                        temperature=temperature,
                        weight=anchor_weight,
                    )

                anchor_losses.append(loss)
                has_active_anchor_pixels = True

        if has_active_anchor_pixels:
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
        )
        total = total + target_total
        stats.update(target_stats)

        _record_stats(history, stats)

    return (
        _weighted_average(out, weight_sum),
        total / max(tile_count, 1),
        _mean_stats(history),
    )


def _decode_tiled_image(
    image: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    tile_size = int(vae.image_size)
    height, width = int(image.shape[0]), int(image.shape[1])
    out = image.new_zeros(image.shape)
    weight_sum = image.new_zeros(image.shape)
    window = _tile_blend_window(
        tile_size,
        tile_overlap,
        reference=image,
    )

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        patch = image[row : row + tile_size, col : col + tile_size].reshape(
            1,
            1,
            tile_size,
            tile_size,
        )
        mu, _ = vae.encode(patch)

        if mu.ndim != 4:
            raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

        validate_finite_tensor("latent", mu)

        decoded = _decode_latent(vae, mu)
        out[row : row + tile_size, col : col + tile_size] = (
            out[row : row + tile_size, col : col + tile_size] + decoded * window
        )
        weight_sum[row : row + tile_size, col : col + tile_size] = (
            weight_sum[row : row + tile_size, col : col + tile_size] + window
        )

    return _weighted_average(out, weight_sum)


def _decode_tiled_image_batch(
    images: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    tile_size = int(vae.image_size)
    batch_size = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    out = images.new_zeros(images.shape)
    weight_sum = images.new_zeros(images.shape)
    window = _tile_blend_window(
        tile_size,
        tile_overlap,
        reference=images,
    )

    for row, col in tile_grid(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
    ):
        patches = images[:, row : row + tile_size, col : col + tile_size].reshape(
            batch_size,
            1,
            tile_size,
            tile_size,
        )
        mu, _ = vae.encode(patches)

        if mu.ndim != 4 or mu.shape[0] != batch_size:
            raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

        validate_finite_tensor("latent", mu)

        decoded = _decode_latent_batch(vae, mu)
        out[:, row : row + tile_size, col : col + tile_size] = (
            out[:, row : row + tile_size, col : col + tile_size] + decoded * window
        )
        weight_sum[:, row : row + tile_size, col : col + tile_size] = (
            weight_sum[:, row : row + tile_size, col : col + tile_size] + window
        )

    return _weighted_average(out, weight_sum)


def _tile_blend_window(
    tile_size: int,
    tile_overlap: int,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if tile_overlap == 0:
        return reference.new_ones((tile_size, tile_size))

    return blend_window(
        tile_size,
        tile_size,
        device=reference.device,
        dtype=reference.dtype,
    )


def _weighted_average(out: torch.Tensor, weight_sum: torch.Tensor) -> torch.Tensor:
    return out / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)


def _record_stats(
    history: dict[str, list[torch.Tensor]],
    stats: dict[str, torch.Tensor],
) -> None:
    for key, value in stats.items():
        history.setdefault(key, []).append(value.detach())


def _mean_stats(history: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack(values).mean()
        for key, values in history.items()
        if values
    }


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
    *,
    steps: int,
    slice_steps: int,
    sds_batch_size: int,
    lr: float,
    slice_schedule: Sequence[tuple[int, int]] | None,
    anchors: Sequence[AnchorSlice] | None,
    anchor_targets: Mapping[tuple[int, int], torch.Tensor] | None,
    anchor_masks: Mapping[tuple[int, int], torch.Tensor] | None,
    anchor_weight: float,
    sds_weight: float,
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
    num_phases: int,
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    validate_floating_dtype("volume dtype", volume.dtype)
    validate_finite_tensor("volume", volume)

    depth, height, width = volume.shape
    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("scale-up SDS requires a cubic volume.")

    _validate_non_negative_integer("steps", steps)
    _validate_non_negative_integer("slice_steps", slice_steps)

    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("num_phases must be an integer.")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    _validate_positive_scalar("lr", lr)
    _validate_positive_scalar("temperature", temperature)

    if not isinstance(sds_batch_size, int) or isinstance(sds_batch_size, bool):
        raise ValueError("sds_batch_size must be an integer.")

    if sds_batch_size <= 0:
        raise ValueError("sds_batch_size must be positive.")

    if slice_schedule is not None and len(slice_schedule) < steps * sds_batch_size:
        raise ValueError("slice_schedule must contain one entry per batched slice.")

    for name, weight in (
        ("sds_weight", sds_weight),
        ("anchor_weight", anchor_weight),
        ("vf_weight", vf_weight),
        ("tpc_weight", tpc_weight),
        ("sa_weight", sa_weight),
        ("diffusivity_weight", diffusivity_weight),
    ):
        _validate_non_negative_scalar(name, weight)

    _validate_anchor_tensor_map(
        "anchor_targets",
        anchor_targets,
        volume_shape=volume.shape,
    )
    _validate_anchor_tensor_map(
        "anchor_masks",
        anchor_masks,
        volume_shape=volume.shape,
        mask=True,
    )

    if anchor_weight > 0.0 and not anchors and not anchor_targets:
        raise ValueError("anchors are required when anchor_weight is positive.")

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

    if diffusivity_weight > 0.0 and diffusivity_solver is None:
        raise ValueError("diffusivity_solver is required for diffusivity loss.")


def _as_anchor_image(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4 and target.shape[:2] == (1, 1):
        return target[0, 0]

    if target.ndim == 2:
        return target

    raise ValueError("anchor target must have shape [H, W] or [1, 1, H, W].")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


def _validate_non_negative_scalar(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")


def _validate_anchor_tensor_map(
    name: str,
    values: Mapping[tuple[int, int], torch.Tensor] | None,
    *,
    volume_shape: torch.Size,
    mask: bool = False,
) -> None:
    if not values:
        return

    for key, value in values.items():
        axis, index = _validate_anchor_tensor_key(
            name,
            key,
            volume_shape=volume_shape,
        )
        image = _as_anchor_image(value)
        if image.shape != _slice_shape(volume_shape, axis):
            raise ValueError(f"{name}[{axis}, {index}] shape must match selected slice.")

        validate_finite_tensor(f"{name}[{axis}, {index}]", image)

        if mask and (image.min().item() < 0.0 or image.max().item() > 1.0):
            raise ValueError(f"{name} values must be between 0 and 1.")


def _validate_anchor_tensor_key(
    name: str,
    key: tuple[int, int],
    *,
    volume_shape: torch.Size,
) -> tuple[int, int]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(f"{name} keys must be (axis, index).")

    axis, index = key
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or not isinstance(index, int)
        or isinstance(index, bool)
    ):
        raise ValueError(f"{name} keys must contain integer axis and index.")

    if axis not in (0, 1, 2):
        raise ValueError(f"{name} axis must be 0, 1, or 2.")

    if index < 0 or index >= volume_shape[axis]:
        raise ValueError(f"{name} index must be inside the selected axis.")

    return axis, index


def _slice_shape(volume_shape: torch.Size, axis: int) -> torch.Size:
    if axis == 0:
        return torch.Size([volume_shape[1], volume_shape[2]])

    if axis == 1:
        return torch.Size([volume_shape[0], volume_shape[2]])

    return torch.Size([volume_shape[0], volume_shape[1]])


def _tensor_map(
    values: Mapping[tuple[int, int], torch.Tensor] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[tuple[int, int], torch.Tensor]:
    if not values:
        return {}

    return {
        (int(axis), int(index)): value.to(device=device, dtype=dtype)
        for (axis, index), value in values.items()
    }
