from collections.abc import Mapping, Sequence

import torch

from src.models import DDPM
from src.predict.slices import extract_slice, replace_slice, select_slice
from src.predict.sds.anchor import anchor_loss, masked_anchor_loss
from src.predict.sds.common import prepare_anchor_targets, prepare_inference_module
from src.predict.sds.core import sds_loss
from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.objective import descriptor_loss
from src.predict.types import AnchorSlice
from src.predict.scale.tiles import tile_grid


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
        lr=lr,
        slice_schedule=slice_schedule,
        anchors=anchors,
        anchor_targets=anchor_targets,
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
    )
    prepared_targets = prepare_anchor_targets(
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=anchor_segment,
        device=volume.device,
        dtype=volume.dtype,
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
        axis, index = select_slice(updated, step, slice_schedule)
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
            image_param.clamp_(-1.0, 1.0)
        _record_stats(history, stats)

    with torch.no_grad():
        decoded = _decode_tiled_image(
            image_param.detach(),
            vae,
            tile_overlap=tile_overlap,
        ).clamp(-1.0, 1.0)
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
    count = image.detach().new_zeros(image.shape)
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
        decoded = _decode_latent(vae, mu)
        out[row : row + tile_size, col : col + tile_size] = (
            out[row : row + tile_size, col : col + tile_size] + decoded
        )
        count[row : row + tile_size, col : col + tile_size] = (
            count[row : row + tile_size, col : col + tile_size] + 1
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

    return out / count.clamp_min(1), total / max(tile_count, 1), _mean_stats(history)


def _decode_tiled_image(
    image: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
) -> torch.Tensor:
    tile_size = int(vae.image_size)
    height, width = int(image.shape[0]), int(image.shape[1])
    out = image.new_zeros(image.shape)
    count = image.new_zeros(image.shape)

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
        decoded = _decode_latent(vae, mu)
        out[row : row + tile_size, col : col + tile_size] = (
            out[row : row + tile_size, col : col + tile_size] + decoded
        )
        count[row : row + tile_size, col : col + tile_size] = (
            count[row : row + tile_size, col : col + tile_size] + 1
        )

    return out / count.clamp_min(1)


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
    return decoded[0, 0]


def _validate_inputs(
    volume: torch.Tensor,
    *,
    steps: int,
    slice_steps: int,
    lr: float,
    slice_schedule: Sequence[tuple[int, int]] | None,
    anchors: Sequence[AnchorSlice] | None,
    anchor_targets: Mapping[tuple[int, int], torch.Tensor] | None,
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
) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    depth, height, width = volume.shape
    if depth != height or depth != width:
        raise ValueError("scale-up SDS requires a cubic volume.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if slice_steps < 0:
        raise ValueError("slice_steps must be non-negative.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if slice_schedule is not None and len(slice_schedule) < steps:
        raise ValueError("slice_schedule must contain at least one entry per step.")
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
