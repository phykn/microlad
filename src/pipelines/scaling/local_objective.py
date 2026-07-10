from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.modeling.diffusion import DDPMProcess
from src.pipelines.guidance.anchor_objective import anchor_loss, masked_anchor_loss
from src.pipelines.guidance.objective import descriptor_loss, descriptor_loss_per_sample
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.pipelines.guidance.prior import sds_loss
from src.pipelines.reconstruction.volume import decode_latent, decode_latents
from src.pipelines.scaling.blending import blend_window
from src.pipelines.scaling.tiles import normalized_tile_weights, tile_grid
from src.pipelines.scaling.validation import _as_anchor_image
from src.common.tensors.validation import validate_finite_tensor

def _local_prior_objective(
    image: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
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
    placements = normalized_tile_weights(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
        device=image.device,
        dtype=image.dtype,
    )
    total = image.sum() * 0.0
    tile_count = 0
    history: dict[str, list[torch.Tensor]] = {}

    for row, col, ownership in placements:
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

        decoded = decode_latent(vae, mu)
        out[row : row + tile_size, col : col + tile_size] = (
            out[row : row + tile_size, col : col + tile_size]
            + decoded * ownership
        )
        weight_sum[row : row + tile_size, col : col + tile_size] = (
            weight_sum[row : row + tile_size, col : col + tile_size] + ownership
        )

        stats: dict[str, torch.Tensor] = {}
        if sds_weight > 0.0:
            latent_weight = F.interpolate(
                ownership.view(1, 1, tile_size, tile_size),
                size=mu.shape[-2:],
                mode="area",
            )[0, 0]
            global_latent_pixels = (
                height
                * width
                * mu.shape[-2]
                * mu.shape[-1]
                / (tile_size * tile_size)
            )
            loss, _ = sds_loss(
                mu,
                diffusion_model,
                ddpm,
                t_min=t_min,
                t_max=t_max,
                spatial_weight=latent_weight,
                spatial_normalizer=global_latent_pixels / len(placements),
            )
            weighted = sds_weight * loss
            total = total + weighted
            stats["sds"] = weighted.detach()

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

    stitched = _weighted_average(out, weight_sum)
    total = total / max(tile_count, 1)
    mean_stats = _mean_stats(history)

    if anchor_weight > 0.0 and anchor_target is not None:
        has_anchor_pixels = (
            anchor_mask is None or bool((anchor_mask > 0).any().item())
        )
        if has_anchor_pixels:
            if anchor_mask is None:
                anchor_total, _ = anchor_loss(
                    stitched,
                    anchor_target,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                )
            else:
                anchor_total, _ = masked_anchor_loss(
                    stitched,
                    anchor_target,
                    anchor_mask,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                )
            total = total + anchor_total
            mean_stats["anchor"] = anchor_total.detach()

    return stitched, total, mean_stats


def _local_prior_objective_batch(
    images: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
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
    placements = normalized_tile_weights(
        height,
        width,
        tile_size=tile_size,
        overlap=tile_overlap,
        device=images.device,
        dtype=images.dtype,
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

    for row, col, ownership in placements:
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

        decoded = decode_latents(vae, mu)
        out[:, row : row + tile_size, col : col + tile_size] = (
            out[:, row : row + tile_size, col : col + tile_size]
            + decoded * ownership
        )
        weight_sum[:, row : row + tile_size, col : col + tile_size] = (
            weight_sum[:, row : row + tile_size, col : col + tile_size]
            + ownership
        )

        stats: dict[str, torch.Tensor] = {}
        if sds_weight > 0.0:
            latent_weight = F.interpolate(
                ownership.view(1, 1, tile_size, tile_size),
                size=mu.shape[-2:],
                mode="area",
            )[0, 0]
            global_latent_pixels = (
                height
                * width
                * mu.shape[-2]
                * mu.shape[-1]
                / (tile_size * tile_size)
            )
            loss, _ = sds_loss(
                mu,
                diffusion_model,
                ddpm,
                t_min=t_min,
                t_max=t_max,
                spatial_weight=latent_weight,
                spatial_normalizer=global_latent_pixels / len(placements),
            )
            weighted = sds_weight * loss
            total = total + weighted
            stats["sds"] = weighted.detach()

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

    stitched = _weighted_average(out, weight_sum)
    total = total / max(tile_count, 1)
    mean_stats = _mean_stats(history)

    anchor_losses = []
    has_active_anchor_pixels = False
    if anchor_weight > 0.0:
        for slice_index, decoded_slice in enumerate(stitched):
            target_image = target_images[slice_index]
            if target_image is None:
                anchor_losses.append(decoded_slice.sum() * 0.0)
                continue

            mask_image = mask_images[slice_index]
            slice_has_anchor_pixels = (
                mask_image is None or bool((mask_image > 0).any().item())
            )
            if not slice_has_anchor_pixels:
                anchor_losses.append(decoded_slice.sum() * 0.0)
                continue

            if mask_image is None:
                anchor_total, _ = anchor_loss(
                    decoded_slice,
                    target_image,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                )
            else:
                anchor_total, _ = masked_anchor_loss(
                    decoded_slice,
                    target_image,
                    mask_image,
                    num_phases=num_phases,
                    temperature=temperature,
                    weight=anchor_weight,
                )
            anchor_losses.append(anchor_total)
            has_active_anchor_pixels = True

    if has_active_anchor_pixels:
        anchor_total = torch.stack(anchor_losses).mean()
        total = total + anchor_total
        mean_stats["anchor"] = anchor_total.detach()

    return stitched, total, mean_stats


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

        decoded = decode_latent(vae, mu)
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

        decoded = decode_latents(vae, mu)
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
