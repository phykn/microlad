import unittest
from unittest.mock import patch

import torch

from src.predict.scale.denoise import denoise_tiled_plane
from src.predict.sampler import DiffusionSampler
from src.predict.scale.sampler import sample_large_lmpdd


class IdentityDDPM:
    num_timesteps = 1

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start


class ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class OrientationDDPM:
    num_timesteps = 3

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        cols = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        return x + (int(t[0].item()) + 1) * (rows * 10 + cols)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start


class BadShapeDDPM:
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], 1, x.shape[-1], device=x.device)


class NonFiniteDDPM:
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float("nan"))


class NonFiniteQSampleDDPM(IdentityDDPM):
    num_timesteps = 2

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_start, float("nan"))


class BroadcastQSampleDDPM(IdentityDDPM):
    num_timesteps = 2

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
        with self.assertRaisesRegex(ValueError, "p_sample"):
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
        with self.assertRaisesRegex(ValueError, "p_sample output.*finite"):
            denoise_tiled_plane(
                ZeroModel(),
                NonFiniteDDPM(),
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, dtype=torch.long),
                tile_size=2,
                overlap=0,
            )


if __name__ == "__main__":
    unittest.main()
