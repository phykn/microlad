def voxel_to_latent_index(slice_index: int, downsample: int = 4) -> int:
    if slice_index < 0:
        raise ValueError("slice_index must be non-negative.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")
    return slice_index // downsample


def tile_starts(length: int, tile_size: int, overlap: int = 0) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size.")
    if tile_size >= length:
        return [0]

    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts
