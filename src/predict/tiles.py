from collections.abc import Iterator

import torch

from ..misc import require_int, require_number


def make_window(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    floor: float = 1e-3,
) -> torch.Tensor:
    require_int("height", height)
    require_int("width", width)
    require_number("floor", floor)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    if floor <= 0.0:
        raise ValueError("floor must be positive.")

    rows = torch.hann_window(
        height,
        periodic=False,
        device=device,
        dtype=dtype,
    )
    cols = torch.hann_window(
        width,
        periodic=False,
        device=device,
        dtype=dtype,
    )
    return torch.outer(rows, cols).clamp_min(floor)


def list_starts(size: int, *, tile_size: int, overlap: int) -> list[int]:
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


def iter_tiles(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
) -> Iterator[tuple[int, int]]:
    for row in list_starts(height, tile_size=tile_size, overlap=overlap):
        for col in list_starts(width, tile_size=tile_size, overlap=overlap):
            yield row, col
