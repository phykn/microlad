import torch


def voxel_to_latent_index(slice_index: int, downsample: int = 4) -> int:
    if slice_index < 0:
        raise ValueError("slice_index must be non-negative.")
    return slice_index // downsample


def insert_condition_slice(
    volume_z: torch.Tensor,
    condition_z: torch.Tensor,
    axis: int,
    slice_index: int,
    downsample: int = 4,
) -> torch.Tensor:
    if volume_z.ndim != 4:
        raise ValueError("volume_z must have shape [C, D, H, W].")
    if condition_z.ndim == 4:
        if condition_z.shape[0] != 1:
            raise ValueError("batched condition_z must have batch size 1.")
        condition_z = condition_z.squeeze(0)
    if condition_z.ndim != 3:
        raise ValueError("condition_z must have shape [C, H, W] or [1, C, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    latent_index = voxel_to_latent_index(slice_index, downsample=downsample)
    result = volume_z.clone()
    if axis == 0:
        result[:, latent_index, :, :] = condition_z
    elif axis == 1:
        result[:, :, latent_index, :] = condition_z
    else:
        result[:, :, :, latent_index] = condition_z
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


def p_sample_conditioned_slice(
    unet: torch.nn.Module,
    ddpm,
    volume_z: torch.Tensor,
    t: torch.Tensor,
    condition_z: torch.Tensor,
    axis: int,
    slice_index: int,
    downsample: int = 4,
) -> torch.Tensor:
    planes = _axis_planes(volume_z, axis)
    batch_t = t.expand(planes.shape[0])
    batch_condition = condition_z.unsqueeze(0).expand(planes.shape[0], -1, -1, -1)
    batch_axis = torch.full((planes.shape[0],), axis, dtype=torch.long, device=volume_z.device)
    batch_slice = torch.full((planes.shape[0],), slice_index, dtype=torch.long, device=volume_z.device)

    pred = unet(planes, batch_t, batch_condition, batch_axis, batch_slice)
    b = batch_t.shape[0]
    coef1 = 1.0 / torch.sqrt(ddpm.alphas[batch_t]).view(b, 1, 1, 1)
    coef2 = ddpm.betas[batch_t].view(b, 1, 1, 1) / ddpm.sqrt_om_acp[batch_t].view(b, 1, 1, 1)
    mean = coef1 * (planes - coef2 * pred)
    noise = torch.randn_like(planes) if (batch_t > 0).any() else torch.zeros_like(planes)
    var = ddpm.posterior_variance[batch_t].view(b, 1, 1, 1)
    next_planes = mean + torch.sqrt(var) * noise
    next_volume = _planes_to_volume(next_planes, axis)
    return insert_condition_slice(next_volume, condition_z, axis, slice_index, downsample=downsample)


def sample_conditioned_latent_volume(
    unet: torch.nn.Module,
    ddpm,
    condition_z: torch.Tensor,
    axis: int,
    slice_index: int,
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16),
    downsample: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    condition_z = condition_z.to(device)
    volume_z = torch.randn(volume_shape, device=device)
    volume_z = insert_condition_slice(volume_z, condition_z, axis, slice_index, downsample=downsample)

    for step in reversed(range(ddpm.num_timesteps)):
        t = torch.tensor([step], dtype=torch.long, device=device)
        volume_z = p_sample_conditioned_slice(
            unet,
            ddpm,
            volume_z,
            t,
            condition_z,
            axis,
            slice_index,
            downsample=downsample,
        )

    return insert_condition_slice(volume_z, condition_z, axis, slice_index, downsample=downsample)
