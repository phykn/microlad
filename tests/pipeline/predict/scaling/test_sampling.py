import unittest
from unittest.mock import patch

import torch

from src.pipeline.predict.scaling.tiles import blend_window
from src.modeling.diffusion import DiffusionSampler
from src.pipeline.predict.scaling.sampling import denoise_tiled_plane, sample_large_lmpdd
from src.pipeline.predict.scaling.tiles import tile_grid


class IdentityDDPM:
    num_timesteps = 1
    posterior_variance = torch.zeros(1)

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class OrientationDDPM:
    num_timesteps = 3
    posterior_variance = torch.zeros(3)

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        cols = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        return x + (int(t[0].item()) + 1) * (rows * 10 + cols)

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.p_mean(model, x, t)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class BadShapeDDPM:
    posterior_variance = torch.zeros(1)

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], 1, x.shape[-1], device=x.device)

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class NonFiniteDDPM:
    posterior_variance = torch.zeros(1)

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float("nan"))

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class MeanOnlyDDPM:
    posterior_variance = torch.tensor([0.0, 4.0])

    def __init__(self) -> None:
        self.calls = 0

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full_like(x, float(self.calls))

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise AssertionError("denoise_tiled_plane must average p_mean, not p_sample.")

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class RecordingBatchDDPM:
    posterior_variance = torch.zeros(1)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        return x

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class NonFiniteQSampleDDPM(IdentityDDPM):
    num_timesteps = 2
    posterior_variance = torch.zeros(2)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_start, float("nan"))


class BroadcastQSampleDDPM(IdentityDDPM):
    num_timesteps = 2
    posterior_variance = torch.zeros(2)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x_start.shape[0],
            1,
            1,
            1,
            dtype=x_start.dtype,
            device=x_start.device,
        )


class ZeroStepDDPM(IdentityDDPM):
    num_timesteps = 0


class ScaleSamplerTest(unittest.TestCase):
    def test_sample_large_lmpdd_injects_anchor_latent(self):
        anchor = torch.zeros(1, 4, 4, 4)
        anchor[:, 2, 1:3, 1:3] = 1
        mask = torch.zeros_like(anchor)
        mask[:, 2, 1:3, 1:3] = 1

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            latent = sample_large_lmpdd(
                ZeroModel(),
                IdentityDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=anchor,
                anchor_mask=mask,
            )

        self.assertTrue(torch.equal(latent[:, 2, 1:3, 1:3], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(latent[:, 0], torch.zeros(1, 4, 4)))

    def test_sample_large_lmpdd_rejects_partial_anchor_inputs(self):
        with self.assertRaisesRegex(ValueError, "anchor_latent and anchor_mask"):
            sample_large_lmpdd(
                ZeroModel(),
                IdentityDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=torch.zeros(1, 4, 4, 4),
            )

    def test_sample_large_lmpdd_rejects_non_integer_latent_shape(self):
        for shape in ((1.5, 4, 4, 4), ("1", 4, 4, 4), (True, 4, 4, 4)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "latent_shape"):
                    sample_large_lmpdd(
                        ZeroModel(),
                        IdentityDDPM(),
                        shape,
                        tile_size=2,
                        tile_overlap=0,
                        device="cpu",
                    )

    def test_sample_large_lmpdd_rejects_invalid_timestep_count(self):
        with self.assertRaisesRegex(ValueError, "num_timesteps"):
            sample_large_lmpdd(
                ZeroModel(),
                ZeroStepDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
            )

    def test_sample_large_lmpdd_rejects_non_finite_anchor_inputs(self):
        cases = [
            (
                torch.full((1, 4, 4, 4), float("inf")),
                torch.ones(1, 4, 4, 4),
            ),
            (
                torch.zeros(1, 4, 4, 4),
                torch.full((1, 4, 4, 4), float("nan")),
            ),
        ]

        for anchor_latent, anchor_mask in cases:
            with self.subTest(anchor_latent=anchor_latent, anchor_mask=anchor_mask):
                with self.assertRaisesRegex(ValueError, "finite"):
                    sample_large_lmpdd(
                        ZeroModel(),
                        IdentityDDPM(),
                        (1, 4, 4, 4),
                        tile_size=2,
                        tile_overlap=0,
                        device="cpu",
                        anchor_latent=anchor_latent,
                        anchor_mask=anchor_mask,
                    )

    def test_sample_large_lmpdd_anchor_blend_keeps_sample_dtype(self):
        anchor = torch.ones(1, 4, 4, 4, dtype=torch.float64)
        mask = torch.ones(1, 4, 4, 4, dtype=torch.float64)

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            latent = sample_large_lmpdd(
                ZeroModel(),
                IdentityDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=anchor,
                anchor_mask=mask,
            )

        self.assertEqual(latent.dtype, torch.float32)
        self.assertTrue(torch.equal(latent, torch.ones(1, 4, 4, 4)))

    def test_sample_large_lmpdd_rejects_non_finite_anchor_noise(self):
        with self.assertRaisesRegex(ValueError, "q_sample output.*finite"):
            sample_large_lmpdd(
                ZeroModel(),
                NonFiniteQSampleDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=torch.zeros(1, 4, 4, 4),
                anchor_mask=torch.ones(1, 4, 4, 4),
            )

    def test_sample_large_lmpdd_rejects_broadcast_anchor_noise(self):
        with self.assertRaisesRegex(ValueError, "q_sample output"):
            sample_large_lmpdd(
                ZeroModel(),
                BroadcastQSampleDDPM(),
                (1, 4, 4, 4),
                tile_size=2,
                tile_overlap=0,
                device="cpu",
                anchor_latent=torch.zeros(1, 4, 4, 4),
                anchor_mask=torch.ones(1, 4, 4, 4),
            )

    def test_sample_large_lmpdd_matches_base_sampler_axis_orientation(self):
        ddpm = OrientationDDPM()
        model = ZeroModel()
        size = 4

        with patch(
            "torch.randn",
            side_effect=[
                torch.zeros(size, 1, size, size),
                torch.zeros(1, size, size, size),
            ],
        ):
            base = DiffusionSampler(model, ddpm, device="cpu").sample_lmpdd(
                (size, 1, size, size)
            )
            large = sample_large_lmpdd(
                model,
                ddpm,
                (1, size, size, size),
                tile_size=size,
                tile_overlap=0,
                device="cpu",
            )

        self.assertTrue(torch.equal(large, base.permute(1, 0, 2, 3).contiguous()))

    def test_denoise_tiled_plane_rejects_bad_sample_shape(self):
        with self.assertRaisesRegex(ValueError, "p_mean"):
            denoise_tiled_plane(
                ZeroModel(),
                BadShapeDDPM(),
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, dtype=torch.long),
                tile_size=2,
                overlap=0,
            )

    def test_denoise_tiled_plane_rejects_non_floating_planes(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            denoise_tiled_plane(
                ZeroModel(),
                IdentityDDPM(),
                torch.zeros(1, 1, 2, 2, dtype=torch.int64),
                torch.zeros(1, dtype=torch.long),
                tile_size=2,
                overlap=0,
            )

    def test_denoise_tiled_plane_rejects_non_finite_planes(self):
        with self.assertRaisesRegex(ValueError, "planes.*finite"):
            denoise_tiled_plane(
                ZeroModel(),
                IdentityDDPM(),
                torch.full((1, 1, 2, 2), float("inf")),
                torch.zeros(1, dtype=torch.long),
                tile_size=2,
                overlap=0,
            )

    def test_denoise_tiled_plane_rejects_non_integer_timesteps(self):
        with self.assertRaisesRegex(ValueError, "timesteps.*integer"):
            denoise_tiled_plane(
                ZeroModel(),
                IdentityDDPM(),
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1),
                tile_size=2,
                overlap=0,
            )

    def test_denoise_tiled_plane_rejects_non_finite_sample_output(self):
        with self.assertRaisesRegex(ValueError, "p_mean output.*finite"):
            denoise_tiled_plane(
                ZeroModel(),
                NonFiniteDDPM(),
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, dtype=torch.long),
                tile_size=2,
                overlap=0,
            )

    def test_denoise_tiled_plane_averages_means_then_adds_one_plane_noise(self):
        ddpm = MeanOnlyDDPM()
        planes = torch.zeros(1, 1, 3, 3)
        timesteps = torch.tensor([1], dtype=torch.long)

        with patch("torch.randn_like", return_value=torch.full_like(planes, 0.25)) as randn:
            denoised = denoise_tiled_plane(
                ZeroModel(),
                ddpm,
                planes,
                timesteps,
                tile_size=2,
                overlap=1,
            )

        expected_mean = torch.tensor(
            [
                [
                    [
                        [1.0, 1.5, 2.0],
                        [2.0, 2.5, 3.0],
                        [3.0, 3.5, 4.0],
                    ]
                ]
            ]
        )
        expected = expected_mean + 0.5

        self.assertTrue(torch.allclose(denoised, expected))
        self.assertEqual(ddpm.calls, 4)
        self.assertEqual(randn.call_count, 1)

    def test_denoise_tiled_plane_weight_blends_means_before_noise(self):
        ddpm = MeanOnlyDDPM()
        planes = torch.zeros(1, 1, 4, 4)
        timesteps = torch.tensor([0], dtype=torch.long)

        with patch("torch.randn_like", return_value=torch.full_like(planes, 99.0)):
            denoised = denoise_tiled_plane(
                ZeroModel(),
                ddpm,
                planes,
                timesteps,
                tile_size=3,
                overlap=2,
            )

        expected = torch.zeros_like(planes)
        weight_sum = torch.zeros_like(planes)
        window = blend_window(
            3,
            3,
            device=planes.device,
            dtype=planes.dtype,
        ).view(1, 1, 3, 3)

        for value, (row, col) in enumerate(
            tile_grid(4, 4, tile_size=3, overlap=2),
            start=1,
        ):
            expected[:, :, row : row + 3, col : col + 3] += value * window
            weight_sum[:, :, row : row + 3, col : col + 3] += window

        expected = expected / weight_sum

        self.assertTrue(torch.allclose(denoised, expected))
        self.assertLess(float(denoised[0, 0, 1, 1]), 2.5)

    def test_denoise_tiled_plane_limits_model_batch_size(self):
        ddpm = RecordingBatchDDPM()
        planes = torch.randn(5, 1, 4, 4)

        denoised = denoise_tiled_plane(
            ZeroModel(),
            ddpm,
            planes,
            torch.zeros(5, dtype=torch.long),
            tile_size=2,
            overlap=0,
            batch_size=2,
        )

        self.assertTrue(torch.equal(denoised, planes))
        self.assertLessEqual(max(ddpm.batch_sizes), 2)


if __name__ == "__main__":
    unittest.main()
