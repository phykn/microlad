import torch

from src.pipelines.scaling.conditioning import center_start
from src.pipelines.scaling.denoising import denoise_tiled_plane
from src.pipelines.scaling.local_objective import _local_prior_objective
from src.pipelines.scaling.tiles import tile_grid
from src.pipelines.reconstruction.slices import extract_slice_batch, replace_slice_batch


class IdentityDDPM:
    posterior_variance = torch.zeros(1)

    def p_mean(
        self,
        model: torch.nn.Module,
        value: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return value

    def _expand(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        ndim: int,
    ) -> torch.Tensor:
        shape = (timestep.shape[0],) + (1,) * (ndim - 1)
        return values[timestep].view(shape)


class IdentityVAE(torch.nn.Module):
    image_size = 3

    def encode(self, image: torch.Tensor):
        return image, torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def test_slice_batch_extract_replace_round_trip_for_every_axis():
    original = torch.arange(
        4 * 5 * 6,
        dtype=torch.float32,
    ).reshape(4, 5, 6)

    for axis in range(3):
        indices = [0, original.shape[axis] - 1]
        selected = extract_slice_batch(original, axis, indices)
        restored = original.clone()

        replace_slice_batch(restored, axis, indices, selected + 1000)

        assert torch.equal(
            extract_slice_batch(restored, axis, indices),
            selected + 1000,
        )


def test_tile_grid_covers_every_pixel():
    coverage = torch.zeros(9, 11, dtype=torch.int64)

    for row, col in tile_grid(9, 11, tile_size=4, overlap=2):
        coverage[row : row + 4, col : col + 4] += 1

    assert torch.all(coverage > 0)


def test_tiled_identity_denoising_matches_input():
    value = torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6)
    timestep = torch.zeros(1, dtype=torch.long)

    for tile_size, overlap in ((6, 0), (4, 2), (3, 1)):
        actual = denoise_tiled_plane(
            torch.nn.Identity(),
            IdentityDDPM(),
            value,
            timestep,
            tile_size=tile_size,
            overlap=overlap,
        )

        assert torch.allclose(actual, value)


def test_center_start_is_symmetric_and_integral():
    assert center_start(volume_size=8, base_size=4) == 2
    assert center_start(volume_size=7, base_size=3) == 2


def test_local_tile_loss_gradient_is_normalized_by_tile_coverage():
    image = torch.full((4, 4), 0.5, requires_grad=True)
    target = torch.zeros(4, 4)

    _, loss, _ = _local_prior_objective(
        image,
        IdentityVAE(),
        torch.nn.Identity(),
        IdentityDDPM(),
        t_min=0,
        t_max=1,
        num_phases=2,
        sds_weight=0.0,
        anchor_target=target,
        anchor_weight=1.0,
        temperature=0.5,
        tile_overlap=2,
    )
    (gradient,) = torch.autograd.grad(loss, image)

    corner = gradient[0, 0].abs()
    center = gradient[1, 1].abs()
    assert corner > 0
    assert torch.allclose(center, corner, atol=1e-6, rtol=1e-6)
