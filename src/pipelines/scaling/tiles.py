import torch

from src.common.validation import require_int
from src.pipelines.scaling.blending import blend_window


def tile_starts(size: int, *, tile_size: int, overlap: int) -> list[int]:
    require_int("size", size)
    require_int("tile_size", tile_size)
    require_int("overlap", overlap)

    if size <= 0:
        raise ValueError("size must be positive.")

    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size.")

    if tile_size > size:
        raise ValueError("tile_size must fit inside size.")

    stride = tile_size - overlap
    starts = list(range(0, size - tile_size + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)

    return starts


def tile_grid(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
):
    for row in tile_starts(height, tile_size=tile_size, overlap=overlap):
        for col in tile_starts(width, tile_size=tile_size, overlap=overlap):
            yield row, col


def normalize_tile_weights(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[int, int, torch.Tensor]]:
    placements = list(
        tile_grid(
            height,
            width,
            tile_size=tile_size,
            overlap=overlap,
        )
    )
    if overlap == 0:
        window = torch.ones(tile_size, tile_size, device=device, dtype=dtype)
    else:
        window = blend_window(
            tile_size,
            tile_size,
            device=device,
            dtype=dtype,
        )

    total = torch.zeros(height, width, device=device, dtype=dtype)
    for row, col in placements:
        total[row : row + tile_size, col : col + tile_size] += window

    return [
        (
            row,
            col,
            window / total[row : row + tile_size, col : col + tile_size],
        )
        for row, col in placements
    ]
