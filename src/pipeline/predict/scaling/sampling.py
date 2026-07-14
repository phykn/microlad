from collections.abc import Sequence

import torch
from tqdm import tqdm

from src.pipeline.predict.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def sample_large_lmpdd(
    model: torch.nn.Module,
    ddpm,
    latent_shape: Sequence[int],
    *,
    tile_size: int,
    tile_overlap: int,
    device: str | torch.device,
    batch_size: int = 16,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    progress: bool = False,
    phase_fractions: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = _validate_latent_shape(latent_shape, tile_size=tile_size)
    num_timesteps = _validate_num_timesteps(ddpm)
    require_int("batch_size", batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean.")
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    latent = torch.randn(shape, device=device)
    anchor_latent, anchor_mask = _prepare_anchor(
        shape,
        device=device,
        dtype=latent.dtype,
        anchor_latent=anchor_latent,
        anchor_mask=anchor_mask,
    )

    steps = tqdm(
        range(num_timesteps - 1, -1, -1),
        total=num_timesteps,
        desc="Scale L-MPDD",
        disable=not progress,
    )
    for pass_index, step in enumerate(steps):
        axis = pass_index % 3
        planes = _lmpdd_pass_to_planes(latent, axis)
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
            batch_size=batch_size,
            phase_fractions=phase_fractions,
        )
        latent = _planes_to_lmpdd_pass(planes, axis)

        if anchor_latent is not None and anchor_mask is not None:
            latent = _blend_anchor(latent, anchor_latent, anchor_mask, ddpm, step)

    return latent


@torch.no_grad()
def denoise_tiled_plane(
    model: torch.nn.Module,
    ddpm,
    planes: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int = 16,
    phase_fractions: torch.Tensor | None = None,
) -> torch.Tensor:
    if planes.ndim != 4:
        raise ValueError("planes must have shape [B, C, H, W].")
    require_float("planes dtype", planes.dtype)
    require_finite("planes", planes)
    require_int("batch_size", batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if timesteps.ndim != 1 or timesteps.shape[0] != planes.shape[0]:
        raise ValueError("timesteps must have shape [B].")
    if timesteps.dtype != torch.long:
        raise ValueError("timesteps must be integer tensors.")

    _, _, height, width = planes.shape
    mean_out = torch.zeros_like(planes)
    weight_sum = torch.zeros(
        (1, 1, height, width),
        dtype=planes.dtype,
        device=planes.device,
    )
    window = (
        torch.ones(
            1,
            1,
            tile_size,
            tile_size,
            dtype=planes.dtype,
            device=planes.device,
        )
        if overlap == 0
        else blend_window(
            tile_size,
            tile_size,
            device=planes.device,
            dtype=planes.dtype,
        ).view(1, 1, tile_size, tile_size)
    )

    for row, col in tile_grid(height, width, tile_size=tile_size, overlap=overlap):
        for start in range(0, planes.shape[0], batch_size):
            stop = min(start + batch_size, planes.shape[0])
            patch = planes[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ]
            if phase_fractions is None:
                mean_tile = ddpm.p_mean(model, patch, timesteps[start:stop])
            else:
                fractions = phase_fractions.to(device=patch.device, dtype=patch.dtype)
                if fractions.ndim == 1:
                    fractions = fractions.unsqueeze(0).expand(patch.shape[0], -1)
                elif fractions.ndim == 2:
                    fractions = fractions[start:stop]
                if fractions.ndim != 2 or fractions.shape[0] != patch.shape[0]:
                    raise ValueError("phase_fractions must have shape [P] or [B, P].")
                mean_tile = ddpm.p_mean(
                    model,
                    patch,
                    timesteps[start:stop],
                    phase_fractions=fractions,
                )
            if mean_tile.shape != patch.shape:
                raise ValueError("ddpm.p_mean output must match input patch shape.")
            require_finite("p_mean output", mean_tile)
            mean_out[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ] += mean_tile * window
        weight_sum[:, :, row : row + tile_size, col : col + tile_size] += window

    mean_out /= weight_sum.clamp_min(torch.finfo(weight_sum.dtype).tiny)
    noise = torch.randn_like(mean_out)
    shape = (timesteps.shape[0],) + (1,) * (mean_out.ndim - 1)
    noise = torch.where(timesteps.view(shape) > 0, noise, torch.zeros_like(noise))
    variance = ddpm._expand(ddpm.posterior_variance, timesteps, mean_out.ndim)
    denoised = mean_out + torch.sqrt(variance) * noise
    require_finite("denoised", denoised)
    return denoised


def _validate_latent_shape(
    latent_shape: Sequence[int],
    *,
    tile_size: int,
) -> tuple[int, int, int, int]:
    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [C, D, H, W].")

    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in latent_shape
    ):
        raise ValueError("latent_shape values must be integers.")

    shape = tuple(latent_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("latent_shape values must be positive.")

    if shape[1] != shape[2] or shape[1] != shape[3]:
        raise ValueError("large L-MPDD sampling requires a cubic latent shape.")

    if int(tile_size) <= 0:
        raise ValueError("tile_size must be positive.")

    if shape[1] < int(tile_size):
        raise ValueError("tile_size must fit inside latent spatial shape.")

    return shape


def _validate_num_timesteps(ddpm) -> int:
    num_timesteps = getattr(ddpm, "num_timesteps", None)
    if not isinstance(num_timesteps, int) or isinstance(num_timesteps, bool):
        raise ValueError("ddpm.num_timesteps must be a positive integer.")

    if num_timesteps <= 0:
        raise ValueError("ddpm.num_timesteps must be a positive integer.")

    return num_timesteps


def _lmpdd_pass_to_planes(latent: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return latent.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return latent.permute(3, 0, 1, 2).contiguous()
    return latent.permute(2, 0, 3, 1).contiguous()


def _planes_to_lmpdd_pass(planes: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return planes.permute(1, 0, 2, 3).contiguous()
    if axis == 1:
        return planes.permute(1, 2, 3, 0).contiguous()
    return planes.permute(1, 3, 0, 2).contiguous()


def _prepare_anchor(
    shape: tuple[int, int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    anchor_latent: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (anchor_latent is None) != (anchor_mask is None):
        raise ValueError("anchor_latent and anchor_mask must be provided together.")

    if anchor_latent is None or anchor_mask is None:
        return None, None

    anchor_latent = anchor_latent.to(device=device, dtype=dtype)
    if anchor_latent.shape != torch.Size(shape):
        raise ValueError("anchor_latent must have the same shape as latent_shape.")

    require_finite("anchor_latent", anchor_latent)

    anchor_mask = anchor_mask.to(device=device, dtype=dtype)
    try:
        anchor_mask = torch.broadcast_to(anchor_mask, anchor_latent.shape)
    except RuntimeError as exc:
        raise ValueError("anchor_mask must be broadcastable to anchor_latent shape.") from exc

    require_finite("anchor_mask", anchor_mask)

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

        if anchor.shape != latent.shape:
            raise ValueError("q_sample output must have the same shape as latent.")

        require_finite("q_sample output", anchor)

    return latent * (1.0 - anchor_mask) + anchor * anchor_mask
