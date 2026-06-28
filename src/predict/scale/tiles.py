def tile_starts(size: int, *, tile_size: int, overlap: int) -> list[int]:
    size = int(size)
    tile_size = int(tile_size)
    overlap = int(overlap)

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
