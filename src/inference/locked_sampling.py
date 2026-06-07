import torch

from .conditions import ConditionLock
from .geometry import tile_starts, voxel_to_latent_index


def _condition_tensor(condition_z: torch.Tensor) -> torch.Tensor:
    if condition_z.ndim == 4:
        if condition_z.shape[0] != 1:
            raise ValueError("batched condition_z must have batch size 1.")
        condition_z = condition_z.squeeze(0)
    if condition_z.ndim != 3:
        raise ValueError("condition_z must have shape [C, H, W] or [1, C, H, W].")
    return condition_z


def _normalized_locks(
    locks: list[ConditionLock],
    device: torch.device,
) -> list[ConditionLock]:
    normalized = []
    for item in locks:
        if not isinstance(item.condition_z, torch.Tensor):
            raise ValueError("lock item must include tensor condition_z.")
        normalized.append(
            ConditionLock(
                condition_z=_condition_tensor(item.condition_z).to(device),
                axis=int(item.axis),
                slice_index=int(item.slice_index),
                row_start=int(item.row_start),
                col_start=int(item.col_start),
                condition_index=int(item.condition_index),
            )
        )
    return normalized


def insert_condition_slice(
    volume_z: torch.Tensor,
    condition_z: torch.Tensor,
    axis: int,
    slice_index: int,
    row_start: int = 0,
    col_start: int = 0,
    strength: float = 1.0,
    downsample: int = 4,
) -> torch.Tensor:
    if volume_z.ndim != 4:
        raise ValueError("volume_z must have shape [C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if strength < 0.0:
        raise ValueError("strength must be non-negative.")

    condition_z = _condition_tensor(condition_z)
    latent_index = voxel_to_latent_index(slice_index, downsample=downsample)
    h, w = condition_z.shape[-2:]
    row_end = row_start + h
    col_end = col_start + w

    result = volume_z.clone()
    if axis == 0:
        current = result[:, latent_index, row_start:row_end, col_start:col_end]
        result[:, latent_index, row_start:row_end, col_start:col_end] = current * (1.0 - strength) + condition_z * strength
    elif axis == 1:
        current = result[:, row_start:row_end, latent_index, col_start:col_end]
        result[:, row_start:row_end, latent_index, col_start:col_end] = current * (1.0 - strength) + condition_z * strength
    else:
        current = result[:, row_start:row_end, col_start:col_end, latent_index]
        result[:, row_start:row_end, col_start:col_end, latent_index] = current * (1.0 - strength) + condition_z * strength
    return result


def apply_condition_locks(
    volume_z: torch.Tensor,
    locks: list[ConditionLock],
    strength: float = 1.0,
    downsample: int = 4,
) -> torch.Tensor:
    result = volume_z
    for item in locks:
        result = insert_condition_slice(
            result,
            condition_z=item.condition_z,
            axis=item.axis,
            slice_index=item.slice_index,
            row_start=item.row_start,
            col_start=item.col_start,
            strength=strength,
            downsample=downsample,
        )
    return result


def _axis_planes(volume_z: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return volume_z.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return volume_z.permute(2, 0, 1, 3).contiguous()
    return volume_z.permute(3, 0, 1, 2).contiguous()


def _planes_to_volume(planes: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return planes.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return planes.permute(1, 2, 0, 3).contiguous()
    return planes.permute(1, 2, 3, 0).contiguous()


def _ddpm_step(unet: torch.nn.Module, ddpm, planes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    batch_t = t.expand(planes.shape[0])
    pred = unet(planes, batch_t)
    b = batch_t.shape[0]
    coef1 = 1.0 / torch.sqrt(ddpm.alphas[batch_t]).view(b, 1, 1, 1)
    coef2 = ddpm.betas[batch_t].view(b, 1, 1, 1) / ddpm.sqrt_om_acp[batch_t].view(b, 1, 1, 1)
    mean = coef1 * (planes - coef2 * pred)
    noise = torch.randn_like(planes) if (batch_t > 0).any() else torch.zeros_like(planes)
    var = ddpm.posterior_variance[batch_t].view(b, 1, 1, 1)
    return mean + torch.sqrt(var) * noise


def denoise_axis(
    unet: torch.nn.Module,
    ddpm,
    volume_z: torch.Tensor,
    t: torch.Tensor,
    axis: int,
    tile_size: int | None = None,
    tile_overlap: int = 0,
) -> torch.Tensor:
    planes = _axis_planes(volume_z, axis)
    if tile_size is None:
        return _planes_to_volume(_ddpm_step(unet, ddpm, planes, t), axis)

    next_planes = torch.zeros_like(planes)
    counts = torch.zeros((1, 1, planes.shape[-2], planes.shape[-1]), device=planes.device, dtype=planes.dtype)
    for row_start in tile_starts(planes.shape[-2], tile_size, tile_overlap):
        for col_start in tile_starts(planes.shape[-1], tile_size, tile_overlap):
            row_end = row_start + tile_size
            col_end = col_start + tile_size
            patch = planes[:, :, row_start:row_end, col_start:col_end]
            next_planes[:, :, row_start:row_end, col_start:col_end] += _ddpm_step(unet, ddpm, patch, t)
            counts[:, :, row_start:row_end, col_start:col_end] += 1
    return _planes_to_volume(next_planes / counts.clamp_min(1), axis)


def sample_locked_latent_volume(
    unet: torch.nn.Module,
    ddpm,
    locks: list[ConditionLock],
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16),
    tile_size: int | None = None,
    tile_overlap: int = 0,
    lock_strength: float = 1.0,
    downsample: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    locks = _normalized_locks(locks, device)
    volume_z = torch.randn(volume_shape, device=device)
    volume_z = apply_condition_locks(volume_z, locks, strength=lock_strength, downsample=downsample)

    for step in reversed(range(ddpm.num_timesteps)):
        t = torch.tensor([step], dtype=torch.long, device=device)
        for axis in (0, 1, 2):
            volume_z = denoise_axis(
                unet=unet,
                ddpm=ddpm,
                volume_z=volume_z,
                t=t,
                axis=axis,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            volume_z = apply_condition_locks(volume_z, locks, strength=lock_strength, downsample=downsample)

    return apply_condition_locks(volume_z, locks, strength=lock_strength, downsample=downsample)
