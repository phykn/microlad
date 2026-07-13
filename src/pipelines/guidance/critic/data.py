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
        variants = []
        for flipped in (False, True):
            for turns in range(4):
                transformed = torch.rot90(image, turns, dims=(-2, -1))
                if flipped:
                    transformed = torch.flip(transformed, dims=(-1,))
                variants.append(transformed)
        augmented.append(torch.stack(variants))

    vae.eval()
    latents = []
    images = torch.stack(augmented).unsqueeze(2).float()
    flat = images.flatten(0, 1)
    for batch in flat.split(batch_size):
        mean, _ = vae.encode(batch)
        latents.append(mean.detach())
    encoded = torch.cat(latents)
    return encoded.unflatten(0, (labels.shape[0], 8))


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
    return torch.unique(torch.cat(values), dim=0)


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


def split_real_bank(
    bank: torch.Tensor,
    *,
    validation_size: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if bank.ndim != 5 or bank.shape[0] <= 0 or bank.shape[1] < 2:
        raise ValueError("real bank must contain sources with multiple augmentations.")
    if bank.shape[0] == 1:
        train, validation = _split_samples(bank[0], validation_size)
        return train, validation, False

    order = torch.randperm(bank.shape[0], device=bank.device)
    validation_sources = min(
        max(1, int(bank.shape[0]) // 4),
        int(bank.shape[0]) - 1,
    )
    validation = bank[order[:validation_sources]].flatten(0, 1)
    train = bank[order[validation_sources:]].flatten(0, 1)
    return train, validation, True


def split_fake_bank(
    volumes: torch.Tensor,
    *,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if volumes.ndim != 5 or volumes.shape[0] < 2:
        raise ValueError("fake bank must contain at least two latent volumes.")
    validation = slice_latents(volumes[:1], crop_size=crop_size)
    train = slice_latents(volumes[1:], crop_size=crop_size)
    return train, validation


def _split_samples(
    bank: torch.Tensor,
    validation_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bank.shape[0] < 2:
        raise ValueError("critic banks must contain at least two samples.")
    order = torch.randperm(bank.shape[0], device=bank.device)
    bank = bank[order]
    validation_size = min(
        max(1, validation_size),
        max(1, int(bank.shape[0]) // 4),
        int(bank.shape[0]) - 1,
    )
    return bank[:-validation_size], bank[-validation_size:]
