import torch

from .slice_conditioned import sample_conditioned_latent_volume, voxel_to_latent_index


@torch.no_grad()
def predict(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    condition: torch.Tensor,
    axis: int,
    slice_index: int,
    condition_is_latent: bool = False,
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16),
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    device = torch.device(device)
    unet.eval()

    if condition_is_latent:
        condition_z = condition.to(device)
        if condition_z.ndim == 4:
            condition_z = condition_z.squeeze(0)
    else:
        vae.eval()
        condition = condition.to(device)
        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        condition_z, _ = vae.encode(condition * 2 - 1)
        condition_z = condition_z.squeeze(0)

    volume_z = sample_conditioned_latent_volume(
        unet=unet,
        ddpm=ddpm,
        condition_z=condition_z,
        axis=axis,
        slice_index=slice_index,
        volume_shape=volume_shape,
        device=device,
    )
    vae.eval()
    volume = vae.decode(volume_z.permute(1, 0, 2, 3))

    latent_index = voxel_to_latent_index(slice_index)
    if axis == 0:
        fixed = volume_z[:, latent_index, :, :]
    elif axis == 1:
        fixed = volume_z[:, :, latent_index, :]
    else:
        fixed = volume_z[:, :, :, latent_index]

    condition_error = float((fixed - condition_z).abs().max())
    return {
        "volume_z": volume_z,
        "volume": volume,
        "condition_z": condition_z,
        "latent_index": latent_index,
        "condition_error": condition_error,
    }


predict_conditioned_volume = predict
