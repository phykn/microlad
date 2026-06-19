import torch
import torch.nn.functional as F

from src.loss.generate import (
    compute_diffusivity_loss,
    compute_gray_moment_loss,
    compute_grayscale_tpc_loss,
    compute_surface_area_loss,
    soft_gray_level_masks,
)

from .condition_stats import ConditionStats, build_condition_stats
from .conditions import FixedSlice
from .decoding import three_axis_refinement


def _slice_shape(volume: torch.Tensor, axis: int) -> tuple[int, int]:
    if axis == 0:
        return int(volume.shape[2]), int(volume.shape[3])
    if axis == 1:
        return int(volume.shape[0]), int(volume.shape[3])
    if axis == 2:
        return int(volume.shape[0]), int(volume.shape[2])
    raise ValueError("axis must be 0, 1, or 2.")


def _validate_slice_index(volume: torch.Tensor, axis: int, index: int) -> None:
    axis_size = (volume.shape[0], volume.shape[2], volume.shape[3])[_axis(axis)]
    if index < 0 or index >= axis_size:
        raise ValueError("slice index must be inside volume.")


def _axis(axis: int) -> int:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return axis


def _select_volume_slice(volume: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    _validate_slice_index(volume, axis, index)
    if axis == 0:
        return volume[index, 0]
    if axis == 1:
        return volume[:, 0, index, :]
    if axis == 2:
        return volume[:, 0, :, index]
    raise ValueError("axis must be 0, 1, or 2.")


def _replace_volume_slice(
    volume: torch.Tensor, axis: int, index: int, image: torch.Tensor
) -> torch.Tensor:
    _validate_slice_index(volume, axis, index)
    if tuple(image.shape) != _slice_shape(volume, axis):
        raise ValueError("slice image shape must match the selected volume plane.")

    result = volume.clone()
    if axis == 0:
        result[index, 0] = image
    elif axis == 1:
        result[:, 0, index, :] = image
    elif axis == 2:
        result[:, 0, :, index] = image
    else:
        raise ValueError("axis must be 0, 1, or 2.")
    return result


def _insert_fixed_slices(
    volume: torch.Tensor,
    fixed_slices: list[FixedSlice] | None,
) -> torch.Tensor:
    if not fixed_slices:
        return volume

    result = volume
    for item in fixed_slices:
        if not isinstance(item.image, torch.Tensor):
            raise ValueError("fixed slice image must be a tensor.")
        image = item.image.to(device=volume.device, dtype=volume.dtype)
        if image.ndim == 3:
            if image.shape[0] != 1:
                raise ValueError(
                    "fixed slice image must have shape [H, W] or [1, H, W]."
                )
            image = image[0]
        if image.ndim != 2:
            raise ValueError("fixed slice image must have shape [H, W] or [1, H, W].")
        result = _replace_volume_slice(result, item.axis, item.index, image)
    return result


def _fixed_slice_image(image: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    image = image.to(device=volume.device, dtype=volume.dtype)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError(
            "condition slice image must have shape [H, W], [1, H, W], or [1, 1, H, W]."
        )
    return image


def _condition_slice_for(
    condition_slices: list[FixedSlice] | None,
    axis: int,
    index: int,
) -> FixedSlice | None:
    if not condition_slices:
        return None
    for item in condition_slices:
        if item.axis == axis and item.index == index:
            return item
    return None


def _validate_sds_slice_args(
    volume: torch.Tensor,
    lr: float,
    t_min: int,
    t_max: int,
    ddpm,
) -> None:
    if volume.ndim != 4:
        raise ValueError("volume must have shape [D, C, H, W].")
    if volume.shape[1] != 1:
        raise ValueError("volume must have a single gray channel.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if t_min < 0 or t_min >= t_max:
        raise ValueError("t_min must be non-negative and smaller than t_max.")
    if t_max > ddpm.num_timesteps:
        raise ValueError("t_max must not exceed ddpm.num_timesteps.")


def _refine_slice(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    axis: int,
    index: int,
    lr: float,
    t_min: int,
    t_max: int,
    stats: ConditionStats,
    stats_weight: float = 0.0,
    diffusivity_weight: float = 0.0,
    condition_image: torch.Tensor | None = None,
    condition_weight: float = 0.0,
    gray_levels: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_sds_slice_args(volume, lr, t_min, t_max, ddpm)

    device = volume.device
    gray_levels = gray_levels or [0, 1]
    with torch.enable_grad():
        image = _select_volume_slice(volume, axis, index).unsqueeze(0).unsqueeze(0)
        mu, _ = vae.encode(image * 2 - 1)
        latent = mu.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=lr)

        t = torch.randint(t_min, t_max, (1,), device=device)
        noise = torch.randn_like(latent)
        alpha = ddpm.sqrt_acp[t].view(1, 1, 1, 1)
        sigma = ddpm.sqrt_om_acp[t].view(1, 1, 1, 1)
        x_t = alpha * latent + sigma * noise
        with torch.no_grad():
            pred = unet(x_t, t)

        target = noise - pred.detach()
        loss_sds = (sigma.pow(2) * (latent * target)).mean()
        decoded = vae.decode(latent).clamp(0, 1)

        if stats_weight > 0 and stats.gray_moments is not None:
            loss_gray_moment = compute_gray_moment_loss(
                decoded,
                target_mean=stats.gray_moments[0],
                target_sqmean=stats.gray_moments[1],
            )
        else:
            loss_gray_moment = torch.tensor(0.0, device=device)

        if (
            stats_weight > 0
            and stats.grayscale_tpc_target is not None
            and stats.grayscale_tpc_bin_mat is not None
            and stats.grayscale_tpc_bin_counts is not None
        ):
            loss_grayscale_tpc = compute_grayscale_tpc_loss(
                decoded,
                stats.grayscale_tpc_target,
                stats.grayscale_tpc_bin_mat,
                stats.grayscale_tpc_bin_counts,
            )
        else:
            loss_grayscale_tpc = torch.tensor(0.0, device=device)

        masks = soft_gray_level_masks(decoded, gray_levels)
        if (
            diffusivity_weight > 0
            and stats.diffusivity_targets is not None
            and stats.diffusivity_solver is not None
        ):
            loss_diffusivity = compute_diffusivity_loss(
                masks,
                stats.diffusivity_solver,
                stats.diffusivity_targets,
                gray_levels,
                device,
            )
        else:
            loss_diffusivity = torch.tensor(0.0, device=device)

        if stats_weight > 0 and stats.surface_area_targets is not None:
            loss_surface_area = compute_surface_area_loss(
                decoded, stats.surface_area_targets, gray_levels, device
            )
        else:
            loss_surface_area = torch.tensor(0.0, device=device)

        if condition_weight > 0 and condition_image is not None:
            condition_target = _fixed_slice_image(condition_image, volume)
            loss_condition = F.mse_loss(decoded, condition_target)
        else:
            loss_condition = torch.tensor(0.0, device=device)

        total = (
            loss_sds
            + stats_weight * loss_gray_moment
            + stats_weight * loss_grayscale_tpc
            + diffusivity_weight * loss_diffusivity
            + stats_weight * loss_surface_area
            + condition_weight * loss_condition
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    with torch.no_grad():
        updated = vae.decode(latent).clamp(0, 1)[0, 0]
        refined = _replace_volume_slice(volume, axis, index, updated)

    return refined, {
        "loss": total.detach(),
        "sds": loss_sds.detach(),
        "gray_moment": loss_gray_moment.detach(),
        "grayscale_tpc": loss_grayscale_tpc.detach(),
        "diffusivity": loss_diffusivity.detach(),
        "surface_area": loss_surface_area.detach(),
        "condition": loss_condition.detach(),
    }


def sds_refine_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    refinement_steps: int = 0,
    condition_images: list[torch.Tensor] | None = None,
    stats_weight: float = 0.0,
    diffusivity_weight: float = 0.0,
    diffusivity_size: int = 32,
    condition_slices: list[FixedSlice] | None = None,
    condition_weight: float = 0.0,
    fixed_slices: list[FixedSlice] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if volume.ndim != 4:
        raise ValueError("volume must have shape [D, C, H, W].")
    if volume.shape[1] != 1:
        raise ValueError("volume must have a single gray channel.")
    if condition_weight > 0 and not condition_slices:
        raise ValueError("condition_slices are required when condition_weight > 0.")

    gray_levels = [0, 1]
    stats = build_condition_stats(
        condition_images=condition_images,
        stats_weight=stats_weight,
        diffusivity_weight=diffusivity_weight,
        diffusivity_size=diffusivity_size,
        gray_levels=gray_levels,
        device=volume.device,
    )

    result = _insert_fixed_slices(volume, fixed_slices)
    depth, _, height, width = result.shape
    axis_sizes = (depth, height, width)
    history = []

    for step in range(steps):
        condition_slice = None
        if condition_weight > 0 and condition_slices:
            condition_slice = condition_slices[step % len(condition_slices)]
            axis = int(condition_slice.axis)
            index = int(condition_slice.index)
        else:
            axis = int(torch.randint(0, 3, (1,), generator=generator).item())
            index = int(
                torch.randint(0, axis_sizes[axis], (1,), generator=generator).item()
            )
            condition_slice = _condition_slice_for(condition_slices, axis, index)
        result, losses = _refine_slice(
            volume=result,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            axis=axis,
            index=index,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            stats=stats,
            stats_weight=stats_weight,
            diffusivity_weight=diffusivity_weight,
            condition_image=condition_slice.image
            if condition_slice is not None
            else None,
            condition_weight=condition_weight,
            gray_levels=gray_levels,
        )
        result = _insert_fixed_slices(result, fixed_slices)
        history.append({name: float(value) for name, value in losses.items()})

    if refinement_steps > 0:
        result = three_axis_refinement(result, vae, refinement_steps=refinement_steps)
        result = _insert_fixed_slices(result, fixed_slices)

    return result, history
