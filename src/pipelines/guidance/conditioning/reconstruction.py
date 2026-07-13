import torch

from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.pipelines.scaling.tiles import blend_window, tile_grid
from src.validation import require_finite


@torch.no_grad()
def reconstruct_target(
    vae: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_overlap: int = 0,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("anchor target image must have shape [1, 1, H, W].")

    require_finite("anchor target image", image)

    num_phases = getattr(vae, "num_phases", None)
    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("vae.num_phases must be an integer.")
    if num_phases < 2:
        raise ValueError("vae.num_phases must be at least 2.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise TypeError("vae must provide decode_probs for categorical reconstruction.")

    image_size = int(vae.image_size)
    height, width = int(image.shape[-2]), int(image.shape[-1])

    if (height, width) == (image_size, image_size):
        return _reconstruct_patch(vae, image, num_phases=num_phases)
    return _reconstruct_tiled_target(
        vae,
        image,
        tile_overlap=tile_overlap,
        num_phases=num_phases,
    )


def _reconstruct_patch(
    vae: torch.nn.Module,
    image: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    vae.eval()
    mu, _ = vae.encode(image)
    require_finite("encoded anchor target", mu)

    probabilities = vae.decode_probs(mu)
    expected_shape = (image.shape[0], num_phases, *image.shape[-2:])
    if probabilities.shape != expected_shape:
        raise ValueError("reconstructed probabilities must match input spatial shape.")
    require_finite("reconstructed anchor probabilities", probabilities)
    return probabilities_to_calibrated_labels(
        probabilities,
        num_phases,
    ).to(image.dtype).detach()


def _reconstruct_tiled_target(
    vae: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_overlap: int,
    num_phases: int,
) -> torch.Tensor:
    image_size = int(vae.image_size)
    height, width = image.shape[-2:]
    out = torch.zeros(
        1,
        num_phases,
        height,
        width,
        dtype=image.dtype,
        device=image.device,
    )
    weight_sum = torch.zeros_like(image)
    window = (
        torch.ones(
            1,
            1,
            image_size,
            image_size,
            dtype=image.dtype,
            device=image.device,
        )
        if tile_overlap == 0
        else blend_window(
            image_size,
            image_size,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, image_size, image_size)
    )

    vae.eval()
    for row, col in tile_grid(
        height,
        width,
        tile_size=image_size,
        overlap=tile_overlap,
    ):
        patch = image[:, :, row : row + image_size, col : col + image_size]
        mu, _ = vae.encode(patch)
        require_finite("encoded anchor target", mu)
        probabilities = vae.decode_probs(mu)
        expected_shape = (1, num_phases, image_size, image_size)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "reconstructed probabilities must match input spatial shape."
            )
        require_finite("reconstructed anchor probabilities", probabilities)
        out[:, :, row : row + image_size, col : col + image_size] += (
            probabilities * window
        )
        weight_sum[:, :, row : row + image_size, col : col + image_size] += window

    probabilities = out / weight_sum.clamp_min(
        torch.finfo(weight_sum.dtype).tiny
    )
    return probabilities_to_calibrated_labels(
        probabilities,
        num_phases,
    ).to(image.dtype).detach()
