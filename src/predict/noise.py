import torch

from ..misc import require_finite
from .tiles import iter_tiles, make_window


def predict_tiles(
    model: torch.nn.Module,
    planes: torch.Tensor,
    steps: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    fractions: torch.Tensor | None,
    guidance: float = 1.0,
) -> torch.Tensor:
    _, _, height, width = planes.shape
    output = torch.zeros_like(planes)
    weights = torch.zeros(
        (1, 1, height, width),
        device=planes.device,
        dtype=planes.dtype,
    )
    window = (
        torch.ones(
            (1, 1, tile_size, tile_size),
            device=planes.device,
            dtype=planes.dtype,
        )
        if overlap == 0
        else make_window(
            tile_size,
            tile_size,
            device=planes.device,
            dtype=planes.dtype,
        ).view(1, 1, tile_size, tile_size)
    )

    for row, col in iter_tiles(
        height,
        width,
        tile_size=tile_size,
        overlap=overlap,
    ):
        for start in range(0, planes.shape[0], batch_size):
            stop = min(start + batch_size, planes.shape[0])
            patch = planes[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ]
            condition = _expand_fractions(fractions, patch.shape[0])
            noise = guide_noise(
                model,
                patch,
                steps[start:stop],
                condition=condition,
                guidance=guidance,
            )
            if noise.shape != patch.shape:
                raise ValueError("model output must have the same shape as its input.")
            require_finite("predicted noise", noise)
            output[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ] += noise * window
        weights[:, :, row : row + tile_size, col : col + tile_size] += window

    return output / weights.clamp_min(torch.finfo(planes.dtype).tiny)


def guide_noise(
    model: torch.nn.Module,
    patch: torch.Tensor,
    steps: torch.Tensor,
    *,
    condition: torch.Tensor | None,
    guidance: float,
) -> torch.Tensor:
    if condition is None:
        return model(patch, steps)
    if guidance == 1.0:
        return model(patch, steps, condition)

    model_input = torch.cat([patch, patch], dim=0)
    model_steps = torch.cat([steps, steps], dim=0)
    model_cond = torch.cat([torch.zeros_like(condition), condition], dim=0)
    null, guided = model(model_input, model_steps, model_cond).chunk(2, dim=0)
    return null + guidance * (guided - null)


def _expand_fractions(
    fractions: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    return None if fractions is None else fractions[None].expand(batch_size, -1)
