import unittest

import numpy as np
import torch

from src.modeling.diffusion import DDPMProcess
from src.app.api import AnchorSlice
from src.pipelines.scaling.tiles import blend_window
from src.pipelines.scaling.optimization import optimize_large_volume
from src.pipelines.scaling.objective import (
    batch_objective,
    slice_objective,
)
from src.pipelines.scaling.tiles import tile_grid


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2

    def encode(self, image: torch.Tensor):
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        base = torch.where(
            latent >= 0.0,
            torch.tanh(latent),
            torch.zeros_like(latent),
        )
        phase_one = 1e-3 + (1.0 - 2e-3) * base
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class ShiftDecodeVAE(IdentityVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        shifted = latent + 0.25
        base = torch.where(
            shifted >= 0.0,
            torch.tanh(shifted),
            torch.zeros_like(shifted),
        )
        phase_one = 1e-3 + (1.0 - 2e-3) * base
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class NonFiniteEncodeVAE(IdentityVAE):
    def encode(self, image: torch.Tensor):
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class NonFiniteDecodeVAE(IdentityVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latent.shape[0], self.num_phases, *latent.shape[-2:]),
            float("nan"),
            dtype=latent.dtype,
            device=latent.device,
        )


class LocalPatternVAE(torch.nn.Module):
    image_size = 3
    latent_size = 3
    latent_ch = 1
    num_phases = 2

    def encode(self, image: torch.Tensor):
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        ).view(1, 1, self.image_size, 1)
        cols = torch.arange(
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        ).view(1, 1, 1, self.image_size)
        phase_one = ((rows * 10.0 + cols) / 22.0).expand(
            latent.shape[0], 1, -1, -1
        )
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class RecordingNoiseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        return torch.zeros_like(x)


class PredictScaleSDSTest(unittest.TestCase):
    def optimize(self, volume: torch.Tensor | None = None, **overrides):
        kwargs = {
            "steps": 1,
            "slice_steps": 1,
            "lr": 0.1,
            "t_min": 1,
            "t_max": 3,
            "num_phases": 2,
            "slice_schedule": [(0, 2)],
            "sds_weight": 0.0,
            "tile_overlap": 0,
        }
        kwargs.update(overrides)

        return optimize_large_volume(
            torch.zeros(4, 4, 4) if volume is None else volume,
            kwargs.pop("vae", IdentityVAE()),
            kwargs.pop("diffusion_model", ZeroNoiseModel()),
            kwargs.pop("ddpm", DDPMProcess(timesteps=4)),
            **kwargs,
        )

    def test_optimize_large_volume_updates_scheduled_anchor_slice_tiles(self):
        volume = torch.zeros(4, 4, 4)
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=2,
        )

        updated, stats = optimize_large_volume(
            volume,
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            slice_steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 2)],
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
            tile_overlap=0,
        )

        self.assertGreater(float(updated[2].mean()), 0.0)
        self.assertLess(float(updated[2].mean()), 1.0)
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertIn("history_anchor", stats)
        self.assertIn("steps", stats)

    def test_optimize_large_volume_with_zero_slice_steps_preserves_slice(self):
        volume = torch.zeros(4, 4, 4)

        updated, stats = optimize_large_volume(
            volume,
            ShiftDecodeVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            slice_steps=0,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 2)],
            sds_weight=0.0,
            tile_overlap=0,
        )

        self.assertTrue(torch.equal(updated, volume))
        self.assertEqual(int(stats["steps"].item()), 1)

    def test_optimize_large_volume_applies_masked_anchor_target_patch(self):
        target = torch.zeros(4, 4)
        target[1:3, 1:3] = 1
        mask = torch.zeros(4, 4)
        mask[1:3, 1:3] = 1

        updated, stats = optimize_large_volume(
            torch.zeros(4, 4, 4),
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            slice_steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 2)],
            anchor_targets={(0, 2): target},
            anchor_masks={(0, 2): mask},
            anchor_weight=1.0,
            sds_weight=0.0,
            tile_overlap=0,
        )

        self.assertTrue(torch.all(updated[2] == updated[2].round()))
        self.assertFalse(torch.equal(updated[2], target))
        self.assertEqual(float(updated[2, 0, 0]), 0.0)
        self.assertIn("history_anchor", stats)

    def test_optimize_large_volume_applies_full_slice_vf_target_loss(self):
        updated, stats = optimize_large_volume(
            torch.zeros(4, 4, 4),
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            slice_steps=2,
            lr=0.5,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 1)],
            sds_weight=0.0,
            vf_targets=torch.tensor([1.0, 0.0]),
            vf_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )

        self.assertLessEqual(float(updated[1].mean()), 0.0)
        self.assertIn("history_vf", stats)
        self.assertIn("history_loss", stats)

    def test_optimize_large_volume_batches_same_axis_slices_for_sds_prior(self):
        model = RecordingNoiseModel()

        updated, stats = optimize_large_volume(
            torch.zeros(4, 4, 4),
            IdentityVAE(),
            model,
            DDPMProcess(timesteps=4),
            steps=1,
            slice_steps=1,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 1), (0, 3)],
            sds_weight=1.0,
            tile_overlap=0,
        )

        self.assertEqual(model.batch_sizes, [2, 2, 2, 2])
        self.assertEqual(updated.shape, torch.Size([4, 4, 4]))
        self.assertIn("history_sds", stats)

    def test_optimize_large_volume_rejects_non_floating_volume(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            self.optimize(torch.zeros(4, 4, 4, dtype=torch.int64))

    def test_optimize_large_volume_rejects_non_finite_volume(self):
        with self.assertRaisesRegex(ValueError, "volume.*finite"):
            self.optimize(torch.full((4, 4, 4), float("nan")))

    def test_optimize_large_volume_rejects_empty_volume(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.optimize(torch.empty(0, 0, 0))

    def test_optimize_large_volume_rejects_invalid_anchor_target_key(self):
        with self.assertRaisesRegex(ValueError, "anchor_targets.*inside"):
            self.optimize(
                anchor_targets={(0, 99): torch.zeros(4, 4)},
                anchor_weight=1.0,
            )

    def test_optimize_large_volume_rejects_non_finite_anchor_mask(self):
        with self.assertRaisesRegex(ValueError, "anchor_masks.*finite"):
            self.optimize(
                anchor_targets={(0, 2): torch.zeros(4, 4)},
                anchor_masks={(0, 2): torch.full((4, 4), float("nan"))},
                anchor_weight=1.0,
            )

    def test_optimize_large_volume_rejects_anchor_mask_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "anchor_masks.*between 0 and 1"):
            self.optimize(
                anchor_targets={(0, 2): torch.zeros(4, 4)},
                anchor_masks={(0, 2): torch.full((4, 4), 2.0)},
                anchor_weight=1.0,
            )

    def test_optimize_large_volume_rejects_non_finite_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "latent.*finite"):
            self.optimize(vae=NonFiniteEncodeVAE())

    def test_optimize_large_volume_rejects_non_finite_decoded_tile(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            self.optimize(vae=NonFiniteDecodeVAE())

    def test_optimize_large_volume_rejects_non_finite_scalar_parameters(self):
        for name in ("lr", "sds_weight", "temperature"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    self.optimize(**{name: float("nan")})

    def test_large_slice_prior_loss_is_averaged_across_tiles(self):
        decoded, _, total, stats = slice_objective(
            torch.zeros(4, 4),
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=torch.ones(4, 4),
            anchor_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )

        self.assertEqual(decoded.shape, torch.Size([4, 4]))
        self.assertIn("anchor", stats)
        self.assertTrue(torch.allclose(total.detach(), stats["anchor"]))

    def test_large_slice_batch_descriptor_loss_averages_per_slice_losses(self):
        images = torch.stack(
            [
                torch.full((2, 2), 0.0),
                torch.full((2, 2), 1.0),
            ]
        )

        _, _, total, stats = batch_objective(
            images,
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_targets=[None, None],
            anchor_masks=[None, None],
            anchor_weight=0.0,
            temperature=0.01,
            tile_overlap=0,
            vf_targets=torch.tensor([0.5, 0.5]),
            vf_weight=1.0,
        )

        self.assertGreater(float(total.detach()), 0.1)
        self.assertGreater(float(stats["vf"]), 0.1)

    def test_large_slice_batch_anchor_loss_includes_unanchored_slices_in_mean(self):
        images = torch.zeros(2, 2, 2)
        target = torch.ones(2, 2)

        _, _, single_total, _ = slice_objective(
            images[0],
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=target,
            anchor_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )
        _, _, batch_total, stats = batch_objective(
            images,
            IdentityVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_targets=[target, None],
            anchor_masks=[None, None],
            anchor_weight=1.0,
            temperature=0.5,
            tile_overlap=0,
        )

        self.assertTrue(torch.allclose(batch_total, single_total / 2.0))
        self.assertTrue(torch.allclose(stats["anchor"], single_total.detach() / 2.0))

    def test_slice_objective_uses_weighted_tile_stitching(self):
        vae = LocalPatternVAE()

        decoded, _, total, stats = slice_objective(
            torch.zeros(4, 4),
            vae,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=None,
            anchor_weight=0.0,
            temperature=0.5,
            tile_overlap=2,
        )

        expected = _expected_local_pattern_stitch(vae, torch.zeros(4, 4))

        self.assertTrue(torch.allclose(decoded, expected))
        self.assertTrue(torch.equal(total.detach(), torch.tensor(0.0)))
        self.assertEqual(stats, {})

    def test_batch_objective_uses_weighted_tile_stitching(self):
        vae = LocalPatternVAE()

        decoded, _, _, _ = batch_objective(
            torch.zeros(2, 4, 4),
            vae,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_targets=[None, None],
            anchor_masks=[None, None],
            anchor_weight=0.0,
            temperature=0.5,
            tile_overlap=2,
        )

        expected = _expected_local_pattern_stitch(vae, torch.zeros(4, 4))

        self.assertTrue(
            torch.allclose(
                decoded,
                expected.unsqueeze(0).expand(2, -1, -1),
            )
        )


def _expected_local_pattern_stitch(
    vae: LocalPatternVAE,
    image: torch.Tensor,
) -> torch.Tensor:
    tile_size = int(vae.image_size)
    out = image.new_zeros(image.shape)
    weight_sum = image.new_zeros(image.shape)
    window = blend_window(
        tile_size,
        tile_size,
        device=image.device,
        dtype=image.dtype,
    )
    pattern = vae.decode_probs(torch.zeros(1, 1, tile_size, tile_size))[0, 1]

    for row, col in tile_grid(
        int(image.shape[0]),
        int(image.shape[1]),
        tile_size=tile_size,
        overlap=2,
    ):
        out[row : row + tile_size, col : col + tile_size] += pattern * window
        weight_sum[row : row + tile_size, col : col + tile_size] += window

    return out / weight_sum


if __name__ == "__main__":
    unittest.main()
