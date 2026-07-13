import torch

from src.modeling.slicegan import (
    SliceGANGenerator,
    slicegan_output_size,
)


@torch.no_grad()
def render_generator_tiled(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    *,
    core_noise_size: int = 4,
    halo_noise_size: int = 4,
    output_device: torch.device | str | None = None,
) -> torch.Tensor:
    """Render a fully convolutional 3D generator in overlapping noise tiles."""

    if generator.training:
        raise ValueError("tiled generator rendering requires eval mode.")
    if isinstance(generator, SliceGANGenerator) and not generator.fully_convolutional:
        raise ValueError("tiled SliceGAN rendering requires fully_convolutional=True.")
    if noise.ndim != 5:
        raise ValueError("noise must have shape [B, C, D, H, W].")
    for name, value in (
        ("core_noise_size", core_noise_size),
        ("halo_noise_size", halo_noise_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
    if core_noise_size <= 0:
        raise ValueError("core_noise_size must be positive.")
    if halo_noise_size < 0:
        raise ValueError("halo_noise_size must be non-negative.")

    output_device = (
        noise.device if output_device is None else torch.device(output_device)
    )
    spatial_shape = tuple(int(size) for size in noise.shape[-3:])
    output_shape = tuple(slicegan_output_size(size) for size in spatial_shape)
    output = None
    for depth_start in range(0, spatial_shape[0], core_noise_size):
        for row_start in range(0, spatial_shape[1], core_noise_size):
            for column_start in range(0, spatial_shape[2], core_noise_size):
                core_starts = (depth_start, row_start, column_start)
                core_stops = tuple(
                    min(size, start + core_noise_size)
                    for size, start in zip(spatial_shape, core_starts, strict=True)
                )
                tile_starts = tuple(
                    max(0, start - halo_noise_size) for start in core_starts
                )
                tile_stops = tuple(
                    min(size, stop + halo_noise_size)
                    for size, stop in zip(spatial_shape, core_stops, strict=True)
                )
                tile_noise = noise[
                    :,
                    :,
                    tile_starts[0] : tile_stops[0],
                    tile_starts[1] : tile_stops[1],
                    tile_starts[2] : tile_stops[2],
                ]
                rendered = generator(tile_noise)
                expected_tile_shape = tuple(
                    slicegan_output_size(stop - start)
                    for start, stop in zip(tile_starts, tile_stops, strict=True)
                )
                if tuple(rendered.shape[-3:]) != expected_tile_shape:
                    raise ValueError(
                        "generator output size must be 16 times its noise-grid size."
                    )
                if output is None:
                    output = torch.empty(
                        noise.shape[0],
                        rendered.shape[1],
                        *output_shape,
                        device=output_device,
                        dtype=rendered.dtype,
                    )
                crop_starts = tuple(
                    16 * (core - tile)
                    for core, tile in zip(core_starts, tile_starts, strict=True)
                )
                crop_stops = tuple(
                    crop_start + slicegan_output_size(core_stop - core_start)
                    for crop_start, core_start, core_stop in zip(
                        crop_starts,
                        core_starts,
                        core_stops,
                        strict=True,
                    )
                )
                output_starts = tuple(16 * start for start in core_starts)
                output_stops = tuple(slicegan_output_size(stop) for stop in core_stops)
                output[
                    :,
                    :,
                    output_starts[0] : output_stops[0],
                    output_starts[1] : output_stops[1],
                    output_starts[2] : output_stops[2],
                ] = rendered[
                    :,
                    :,
                    crop_starts[0] : crop_stops[0],
                    crop_starts[1] : crop_stops[1],
                    crop_starts[2] : crop_stops[2],
                ].to(output_device)
    if output is None:
        raise RuntimeError("tiled generator rendering produced no tiles.")
    return output
