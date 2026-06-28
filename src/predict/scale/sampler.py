from collections.abc import Sequence

import torch

from src.predict.scale.denoise import denoise_tiled_plane


@torch.no_grad()
def sample_large_lmpdd(
    model: torch.nn.Module,
    ddpm,
    latent_shape: Sequence[int],
    *,
    tile_size: int,
    tile_overlap: int,
    device: str | torch.device,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = _validate_latent_shape(latent_shape, tile_size=tile_size)
    device = torch.device(device)
    anchor_latent, anchor_mask = _prepare_anchor(
        shape,
        device=device,
        anchor_latent=anchor_latent,
        anchor_mask=anchor_mask,
    )

    model = model.to(device)
    model.eval()

    latent = torch.randn(shape, device=device)
    for pass_index, step in enumerate(range(int(ddpm.num_timesteps) - 1, -1, -1)):
        axis = pass_index % 3
        planes = _axis_to_planes(latent, axis)
        timesteps = torch.full(
            (planes.shape[0],),
            step,
            dtype=torch.long,
            device=planes.device,
        )
        planes = denoise_tiled_plane(
            model,
            ddpm,
            planes,
            timesteps,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
        latent = _planes_to_axis(planes, axis)
        if anchor_latent is not None and anchor_mask is not None:
            latent = _blend_anchor(latent, anchor_latent, anchor_mask, ddpm, step)

    return latent


def _validate_latent_shape(
    latent_shape: Sequence[int],
    *,
    tile_size: int,
) -> tuple[int, int, int, int]:
    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [C, D, H, W].")
    shape = tuple(int(value) for value in latent_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("latent_shape values must be positive.")
    if shape[1] != shape[2] or shape[1] != shape[3]:
        raise ValueError("large L-MPDD sampling requires a cubic latent shape.")
    if int(tile_size) <= 0:
        raise ValueError("tile_size must be positive.")
    if shape[1] < int(tile_size):
        raise ValueError("tile_size must fit inside latent spatial shape.")
    return shape


def _axis_to_planes(latent: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return latent.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return latent.permute(2, 0, 1, 3).contiguous()
    return latent.permute(3, 0, 1, 2).contiguous()


def _planes_to_axis(planes: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return planes.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return planes.permute(1, 2, 0, 3).contiguous()
    return planes.permute(1, 2, 3, 0).contiguous()


def _prepare_anchor(
    shape: tuple[int, int, int, int],
    *,
    device: torch.device,
    anchor_latent: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (anchor_latent is None) != (anchor_mask is None):
        raise ValueError("anchor_latent and anchor_mask must be provided together.")
    if anchor_latent is None or anchor_mask is None:
        return None, None

    anchor_latent = anchor_latent.to(device=device)
    if anchor_latent.shape != torch.Size(shape):
        raise ValueError("anchor_latent must have the same shape as latent_shape.")

    anchor_mask = anchor_mask.to(device=device, dtype=anchor_latent.dtype)
    try:
        anchor_mask = torch.broadcast_to(anchor_mask, anchor_latent.shape)
    except RuntimeError as exc:
        raise ValueError("anchor_mask must be broadcastable to anchor_latent shape.") from exc
    if anchor_mask.min().item() < 0.0 or anchor_mask.max().item() > 1.0:
        raise ValueError("anchor_mask values must be between 0 and 1.")
    return anchor_latent, anchor_mask


def _blend_anchor(
    latent: torch.Tensor,
    anchor_latent: torch.Tensor,
    anchor_mask: torch.Tensor,
    ddpm,
    step: int,
) -> torch.Tensor:
    if step == 0:
        anchor = anchor_latent
    else:
        t = torch.full(
            (anchor_latent.shape[0],),
            step - 1,
            dtype=torch.long,
            device=latent.device,
        )
        anchor = ddpm.q_sample(anchor_latent, t)
    return latent * (1.0 - anchor_mask) + anchor * anchor_mask
