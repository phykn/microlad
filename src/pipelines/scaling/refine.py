from collections.abc import Sequence

import torch

from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.pipelines.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def refine_large_candidates(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    candidates: Sequence[int],
    tile_overlap: int = 0,
    tile_batch_size: int = 16,
) -> tuple[torch.Tensor, ...]:
    if not candidates:
        raise ValueError("candidates must not be empty.")
    for steps in candidates:
        require_int("candidate steps", steps)
        if steps < 0:
            raise ValueError("candidate steps must be non-negative.")
    require_int("tile_batch_size", tile_batch_size)
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be positive.")
    _validate_volume(volume)

    num_phases = getattr(vae, "num_phases", None)
    require_int("vae.num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("vae.num_phases must be at least 2.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise ValueError("large volume refinement requires vae.decode_probs.")

    refined = volume.float()
    selected = {0: refined.clone()} if 0 in candidates else {}
    vae.eval()
    for step in range(1, max(candidates) + 1):
        refined = _refine_once(
            refined,
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            num_phases=num_phases,
        )
        if step in candidates:
            selected[step] = refined.clone()
    return tuple(selected[steps] for steps in candidates)


def _refine_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int,
    num_phases: int,
) -> torch.Tensor:
    depth, height, width = volume.shape
    log_sum = torch.zeros(
        num_phases,
        depth,
        height,
        width,
        dtype=torch.float32,
        device=volume.device,
    )

    for index in range(depth):
        probabilities = _refine_tiled_plane_probabilities(
            volume[index, :, :],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            num_phases=num_phases,
        )
        log_sum[:, index, :, :] += probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()

    for index in range(height):
        probabilities = _refine_tiled_plane_probabilities(
            volume[:, index, :],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            num_phases=num_phases,
        )
        log_sum[:, :, index, :] += probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()

    for index in range(width):
        probabilities = _refine_tiled_plane_probabilities(
            volume[:, :, index],
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            num_phases=num_phases,
        )
        log_sum[:, :, :, index] += probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()

    log_sum.div_(3.0)
    log_sum.sub_(log_sum.amax(dim=0, keepdim=True))
    log_sum.exp_()
    log_sum.div_(log_sum.sum(dim=0, keepdim=True))
    return probabilities_to_calibrated_labels(
        log_sum.unsqueeze(0),
        num_phases,
    )[0, 0].float()


def _refine_tiled_plane_probabilities(
    image: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int,
    num_phases: int,
) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")

    tile_size = int(vae.image_size)
    height, width = map(int, image.shape)
    out = torch.zeros(
        num_phases,
        height,
        width,
        dtype=torch.float32,
        device=image.device,
    )
    weight_sum = torch.zeros(height, width, dtype=torch.float32, device=image.device)
    window = (
        torch.ones(tile_size, tile_size, dtype=out.dtype, device=out.device)
        if tile_overlap == 0
        else blend_window(
            tile_size,
            tile_size,
            device=out.device,
            dtype=out.dtype,
        )
    )
    positions = list(
        tile_grid(
            height,
            width,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
    )

    for start in range(0, len(positions), tile_batch_size):
        chunk = positions[start : start + tile_batch_size]
        batch = torch.stack(
            [
                image[row : row + tile_size, col : col + tile_size]
                for row, col in chunk
            ],
            dim=0,
        ).view(len(chunk), 1, tile_size, tile_size)
        mu, _ = vae.encode(batch)
        if mu.ndim != 4 or mu.shape[0] != len(chunk):
            raise ValueError("encode output must have shape [B, C, H, W].")
        require_finite("encoded latent", mu)

        decoded = vae.decode_probs(mu)
        expected = (len(chunk), num_phases, tile_size, tile_size)
        if decoded.shape != expected:
            raise ValueError(
                "decode_probs output must have shape [B, num_phases, H, W]."
            )
        require_finite("decoded probabilities", decoded)

        for tile, (row, col) in zip(decoded, chunk, strict=True):
            out[:, row : row + tile_size, col : col + tile_size] += (
                tile.float() * window.unsqueeze(0)
            )
            weight_sum[row : row + tile_size, col : col + tile_size] += window

    return out / weight_sum.clamp_min(
        torch.finfo(weight_sum.dtype).tiny
    ).unsqueeze(0)


def _validate_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    require_float("volume dtype", volume.dtype)
    require_finite("volume", volume)
    depth, height, width = volume.shape
    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")
    if depth != height or depth != width:
        raise ValueError("large volume refinement requires a cubic volume.")
