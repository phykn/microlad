import torch


@torch.no_grad()
def encode_refs(
    vae: torch.nn.Module,
    labels: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if labels.ndim != 3:
        raise ValueError("reference labels must have shape [B, H, W].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    augmented = []
    for image in labels:
        for flipped in (False, True):
            for turns in range(4):
                transformed = torch.rot90(image, turns, dims=(-2, -1))
                if flipped:
                    transformed = torch.flip(transformed, dims=(-1,))
                augmented.append(transformed)

    vae.eval()
    latents = []
    images = torch.stack(augmented).unsqueeze(1).float()
    for batch in images.split(batch_size):
        mean, _ = vae.encode(batch)
        latents.append(mean.detach())
    return torch.cat(latents)


def merge_refs(
    anchors: torch.Tensor | None,
    targets: torch.Tensor | None,
) -> torch.Tensor | None:
    values = [value for value in (anchors, targets) if value is not None]
    if not values:
        return None
    size = tuple(values[0].shape[-2:])
    if any(tuple(value.shape[-2:]) != size for value in values):
        raise ValueError("critic references must have the same image size.")
    return torch.cat(values)


def slice_latents(
    volumes: torch.Tensor,
    *,
    crop_size: int,
) -> torch.Tensor:
    if volumes.ndim != 5:
        raise ValueError("latent volumes must have shape [N, C, D, H, W].")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    planes = []
    for volume in volumes:
        planes.extend(
            (
                volume.permute(1, 0, 2, 3),
                volume.permute(2, 0, 1, 3),
                volume.permute(3, 0, 1, 2),
            )
        )
    bank = torch.cat(planes)
    if min(bank.shape[-2:]) < crop_size:
        raise ValueError("fake latent slices are smaller than real references.")
    row = (int(bank.shape[-2]) - crop_size) // 2
    col = (int(bank.shape[-1]) - crop_size) // 2
    return bank[:, :, row : row + crop_size, col : col + crop_size].detach()
