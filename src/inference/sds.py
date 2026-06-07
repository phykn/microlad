import torch
import torch.nn.functional as F

from src.loss import (
    compute_diffusivity_loss,
    compute_grayscale_tpc_loss,
    compute_sa_loss,
    compute_tpc_loss_ste,
    compute_vf_loss,
    compute_vf_moment_loss,
)

from .conditions import FixedSlice
from .decoding import three_axis_refinement


def _select_volume_slice(volume: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    if axis == 0:
        return volume[index, 0]
    if axis == 1:
        return volume[:, 0, index, :]
    if axis == 2:
        return volume[:, 0, :, index]
    raise ValueError("axis must be 0, 1, or 2.")


def _replace_volume_slice(volume: torch.Tensor, axis: int, index: int, image: torch.Tensor) -> torch.Tensor:
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
                raise ValueError("fixed slice image must have shape [H, W] or [1, H, W].")
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
        raise ValueError("condition slice image must have shape [H, W], [1, H, W], or [1, 1, H, W].")
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


def _soft_phase_masks(decoded: torch.Tensor, phases: list[int], beta: float = 30.0) -> torch.Tensor:
    phase_count = len(phases)
    levels = torch.linspace(0.0, 1.0, phase_count, device=decoded.device, dtype=decoded.dtype)
    x = decoded.repeat(1, phase_count, 1, 1)
    dist = torch.abs(x - levels.view(1, phase_count, 1, 1)).view(1, phase_count, -1)
    return F.softmax(-beta * dist, dim=1).view_as(x)


def sds_refine_slice(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    axis: int,
    index: int,
    lr: float,
    t_min: int,
    t_max: int,
    phases: list[int] | None = None,
    vf_targets: tuple[float, float, float] | None = None,
    vf_moments: tuple[float, float] | None = None,
    vf_weight: float = 0.0,
    tpc_targets: dict[int, torch.Tensor] | None = None,
    bin_mat: torch.Tensor | None = None,
    bin_counts: torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    grayscale_tpc_target: torch.Tensor | None = None,
    grayscale_tpc_bin_mat: torch.Tensor | None = None,
    grayscale_tpc_bin_counts: torch.Tensor | None = None,
    grayscale_tpc_weight: float = 0.0,
    rd_targets: dict[int, float] | None = None,
    fem_solver: torch.nn.Module | None = None,
    rd_weight: float = 0.0,
    sa_targets: dict[int, float] | None = None,
    sa_weight: float = 0.0,
    condition_image: torch.Tensor | None = None,
    condition_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if volume.ndim != 4:
        raise ValueError("volume must have shape [D, C, H, W].")

    device = volume.device
    phases = phases or [0, 1]
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

        if vf_weight > 0 and vf_moments is not None:
            loss_vf = compute_vf_moment_loss(
                decoded,
                target_mean=vf_moments[0],
                target_sqmean=vf_moments[1],
            )
        elif vf_weight > 0 and vf_targets is not None:
            loss_vf = compute_vf_loss(
                decoded,
                vf0=vf_targets[0],
                vf05=vf_targets[1],
                vf1=vf_targets[2],
            )
        else:
            loss_vf = torch.tensor(0.0, device=device)

        masks = _soft_phase_masks(decoded, phases)
        if tpc_weight > 0 and tpc_targets is not None and bin_mat is not None and bin_counts is not None:
            loss_tpc = compute_tpc_loss_ste(masks[0], phases, tpc_targets, bin_mat, bin_counts, device)
        else:
            loss_tpc = torch.tensor(0.0, device=device)

        if (
            grayscale_tpc_weight > 0
            and grayscale_tpc_target is not None
            and grayscale_tpc_bin_mat is not None
            and grayscale_tpc_bin_counts is not None
        ):
            loss_grayscale_tpc = compute_grayscale_tpc_loss(
                decoded,
                grayscale_tpc_target,
                grayscale_tpc_bin_mat,
                grayscale_tpc_bin_counts,
            )
        else:
            loss_grayscale_tpc = torch.tensor(0.0, device=device)

        if rd_weight > 0 and rd_targets is not None and fem_solver is not None:
            loss_rd = compute_diffusivity_loss(masks, fem_solver, rd_targets, phases, device)
        else:
            loss_rd = torch.tensor(0.0, device=device)

        if sa_weight > 0 and sa_targets is not None:
            loss_sa = compute_sa_loss(decoded, sa_targets, phases, device)
        else:
            loss_sa = torch.tensor(0.0, device=device)

        if condition_weight > 0 and condition_image is not None:
            condition_target = _fixed_slice_image(condition_image, volume)
            loss_condition = F.mse_loss(decoded, condition_target)
        else:
            loss_condition = torch.tensor(0.0, device=device)

        total = (
            loss_sds
            + vf_weight * loss_vf
            + tpc_weight * loss_tpc
            + grayscale_tpc_weight * loss_grayscale_tpc
            + rd_weight * loss_rd
            + sa_weight * loss_sa
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
        "vf": loss_vf.detach(),
        "tpc": loss_tpc.detach(),
        "grayscale_tpc": loss_grayscale_tpc.detach(),
        "rd": loss_rd.detach(),
        "sa": loss_sa.detach(),
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
    phases: list[int] | None = None,
    vf_targets: tuple[float, float, float] | None = None,
    vf_moments: tuple[float, float] | None = None,
    vf_weight: float = 0.0,
    tpc_targets: dict[int, torch.Tensor] | None = None,
    bin_mat: torch.Tensor | None = None,
    bin_counts: torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    grayscale_tpc_target: torch.Tensor | None = None,
    grayscale_tpc_bin_mat: torch.Tensor | None = None,
    grayscale_tpc_bin_counts: torch.Tensor | None = None,
    grayscale_tpc_weight: float = 0.0,
    rd_targets: dict[int, float] | None = None,
    fem_solver: torch.nn.Module | None = None,
    rd_weight: float = 0.0,
    sa_targets: dict[int, float] | None = None,
    sa_weight: float = 0.0,
    condition_slices: list[FixedSlice] | None = None,
    condition_weight: float = 0.0,
    fixed_slices: list[FixedSlice] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if volume.ndim != 4:
        raise ValueError("volume must have shape [D, C, H, W].")

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
            index = int(torch.randint(0, axis_sizes[axis], (1,), generator=generator).item())
            condition_slice = _condition_slice_for(condition_slices, axis, index)
        result, losses = sds_refine_slice(
            volume=result,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            axis=axis,
            index=index,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            phases=phases,
            vf_targets=vf_targets,
            vf_moments=vf_moments,
            vf_weight=vf_weight,
            tpc_targets=tpc_targets,
            bin_mat=bin_mat,
            bin_counts=bin_counts,
            tpc_weight=tpc_weight,
            grayscale_tpc_target=grayscale_tpc_target,
            grayscale_tpc_bin_mat=grayscale_tpc_bin_mat,
            grayscale_tpc_bin_counts=grayscale_tpc_bin_counts,
            grayscale_tpc_weight=grayscale_tpc_weight,
            rd_targets=rd_targets,
            fem_solver=fem_solver,
            rd_weight=rd_weight,
            sa_targets=sa_targets,
            sa_weight=sa_weight,
            condition_image=condition_slice.image if condition_slice is not None else None,
            condition_weight=condition_weight,
        )
        result = _insert_fixed_slices(result, fixed_slices)
        history.append({name: float(value) for name, value in losses.items()})

    if refinement_steps > 0:
        result = three_axis_refinement(result, vae, refinement_steps=refinement_steps)
        result = _insert_fixed_slices(result, fixed_slices)

    return result, history
