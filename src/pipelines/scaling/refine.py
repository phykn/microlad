from collections.abc import Sequence

import torch

from src.pipelines.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def refine_large_probabilities(
    probabilities: torch.Tensor,
    vae: torch.nn.Module,
    *,
    candidates: Sequence[int],
    tile_overlap: int = 0,
    tile_batch_size: int | None = 16,
    strength: float = 0.15,
    anchor_strength: float = 0.05,
    anchor_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if not candidates:
        raise ValueError("candidates must not be empty.")
    for steps in candidates:
        require_int("candidate steps", steps)
        if steps < 0:
            raise ValueError("candidate steps must be non-negative.")
    if tile_batch_size is not None:
        require_int("tile_batch_size", tile_batch_size)
        if tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive or None.")
    for name, value in (("strength", strength), ("anchor_strength", anchor_strength)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    _validate_probabilities(probabilities, vae)

    if anchor_mask is not None:
        expected = (1, 1, *probabilities.shape[2:])
        if anchor_mask.shape != expected or anchor_mask.dtype != torch.bool:
            raise ValueError(f"anchor_mask must be boolean with shape {expected}.")
        anchor_mask = anchor_mask.to(device=probabilities.device)

    normalized = probabilities.float() / probabilities.sum(dim=1, keepdim=True)
    selected = {0: normalized.clone()} if 0 in candidates else {}
    vae.eval()
    for step in range(1, max(candidates) + 1):
        labels = normalized.argmax(dim=1)[0].float()
        projected = _project_once(
            labels,
            vae,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            num_phases=int(probabilities.shape[1]),
        )
        blend = torch.full_like(normalized[:, :1], strength)
        if anchor_mask is not None:
            blend = torch.where(
                anchor_mask,
                blend.new_full((), anchor_strength),
                blend,
            )
        tiny = torch.finfo(normalized.dtype).tiny
        logits = (
            normalized.clamp_min(tiny).log() * (1.0 - blend)
            + projected.clamp_min(tiny).log() * blend
        )
        normalized = logits.softmax(dim=1)
        if step in candidates:
            selected[step] = normalized.clone()
    return tuple(selected[steps] for steps in candidates)


def _project_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int | None,
    num_phases: int,
) -> torch.Tensor:
    oriented = (
        volume,
        volume.permute(1, 0, 2).contiguous(),
        volume.permute(2, 0, 1).contiguous(),
    )
    depth = _project_axis(
        oriented[0],
        vae,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        num_phases=num_phases,
    ).permute(1, 0, 2, 3)
    height = _project_axis(
        oriented[1],
        vae,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        num_phases=num_phases,
    ).permute(1, 2, 0, 3)
    width = _project_axis(
        oriented[2],
        vae,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        num_phases=num_phases,
    ).permute(1, 2, 3, 0)
    tiny = torch.finfo(depth.dtype).tiny
    logits = torch.stack([depth, height, width]).clamp_min(tiny).log().mean(dim=0)
    return logits.softmax(dim=0).unsqueeze(0)


def _project_axis(
    planes: torch.Tensor,
    vae: torch.nn.Module,
    *,
    tile_overlap: int,
    tile_batch_size: int | None,
    num_phases: int,
) -> torch.Tensor:
    tile_size = int(vae.image_size)
    plane_count, height, width = map(int, planes.shape)
    out = torch.zeros(
        plane_count,
        num_phases,
        height,
        width,
        dtype=torch.float32,
        device=planes.device,
    )
    weight_sum = torch.zeros(height, width, dtype=torch.float32, device=planes.device)
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
    jobs = [
        (plane, row, col)
        for plane in range(plane_count)
        for row, col in positions
    ]
    chunk_size = len(jobs) if tile_batch_size is None else tile_batch_size
    for row, col in positions:
        weight_sum[row : row + tile_size, col : col + tile_size] += window

    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start : start + chunk_size]
        batch = torch.stack(
            [
                planes[plane, row : row + tile_size, col : col + tile_size]
                for plane, row, col in chunk
            ]
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

        for tile, (plane, row, col) in zip(decoded, chunk, strict=True):
            out[plane, :, row : row + tile_size, col : col + tile_size] += (
                tile.float() * window.unsqueeze(0)
            )

    return out / weight_sum.clamp_min(
        torch.finfo(weight_sum.dtype).tiny
    ).view(1, 1, height, width)


def _validate_probabilities(
    probabilities: torch.Tensor,
    vae: torch.nn.Module,
) -> None:
    num_phases = getattr(vae, "num_phases", None)
    require_int("vae.num_phases", num_phases)
    expected_prefix = (1, num_phases)
    if probabilities.ndim != 5 or probabilities.shape[:2] != expected_prefix:
        raise ValueError("probabilities must have shape [1, P, D, H, W].")
    if len(set(map(int, probabilities.shape[2:]))) != 1:
        raise ValueError("large probability volume must be cubic.")
    if min(map(int, probabilities.shape[2:])) < int(vae.image_size):
        raise ValueError("large probability volume must contain a trained-size tile.")
    require_float("probabilities dtype", probabilities.dtype)
    require_finite("probabilities", probabilities)
    if torch.any(probabilities < 0.0) or torch.any(
        probabilities.sum(dim=1) <= 0.0
    ):
        raise ValueError("probabilities must contain positive phase mass.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise ValueError("large refinement requires vae.decode_probs.")
