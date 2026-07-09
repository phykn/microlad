import unittest

import numpy as np
import torch
from torch import nn

from src.models import DDPM
from src.predict import AnchorSlice
from src.predict.sds import DiffusivitySolver, optimize_slice, optimize_volume
from src.predict.sds.optimize import _objective, _objective_batch


class IdentityVAE(nn.Module):
    image_size = 4
    latent_size = 4
    latent_ch = 1

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.clone(), torch.zeros_like(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class NonFiniteEncodeVAE(IdentityVAE):
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(x, float("nan")), torch.zeros_like(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z)


class NonFiniteFinalDecodeVAE(IdentityVAE):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1

        if self.decode_calls > 1:
            return torch.full_like(z, float("nan"))

        return z


class ZeroNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class RecordingNoiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        return torch.zeros_like(x)


class PredictSDSOptimizeTest(unittest.TestCase):
    def test_optimize_slice_updates_only_selected_slice(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPM(timesteps=4)

        updated, stats = optimize_slice(
            volume,
            vae,
            model,
            ddpm,
            axis=0,
            index=1,
            steps=4,
            lr=0.5,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            vf_targets={0: 1.0, 1: 0.0},
            vf_weight=1.0,
            temperature=0.5,
        )

        self.assertTrue(torch.allclose(volume, torch.zeros_like(volume)))
        self.assertLess(float(updated[1].mean()), 0.0)
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertTrue(torch.allclose(updated[2], volume[2]))
        self.assertTrue(torch.allclose(updated[3], volume[3]))
        self.assertIn("loss", stats)
        self.assertIn("vf", stats)

    def test_optimize_slice_uses_soft_anchor_loss_without_forced_overwrite(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPM(timesteps=4)
        anchor_target = torch.ones(1, 1, 4, 4)

        updated, stats = optimize_slice(
            volume,
            vae,
            model,
            ddpm,
            axis=1,
            index=2,
            steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=anchor_target,
            anchor_weight=1.0,
        )

        self.assertGreater(float(updated[:, 2, :].mean()), 0.0)
        self.assertLess(float(updated[:, 2, :].mean()), 1.0)
        self.assertIn("anchor", stats)
        self.assertTrue(torch.allclose(updated[:, 0, :], volume[:, 0, :]))

    def test_optimize_slice_combines_all_sds_objectives(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPM(timesteps=4)
        diffusivity_solver = DiffusivitySolver(height=2, width=2, low_cond=0.1)

        updated, stats = optimize_slice(
            volume,
            vae,
            model,
            ddpm,
            axis=2,
            index=0,
            steps=1,
            lr=0.01,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=1.0,
            anchor_target=torch.zeros(4, 4),
            anchor_weight=1.0,
            vf_targets=torch.tensor([0.5, 0.5]),
            vf_weight=1.0,
            tpc_targets=torch.zeros(2, 4),
            tpc_weight=1.0,
            sa_targets=torch.zeros(2),
            sa_weight=1.0,
            diffusivity_targets=torch.zeros(2),
            diffusivity_solver=diffusivity_solver,
            diffusivity_weight=1.0,
        )

        self.assertEqual(updated.shape, volume.shape)
        for key in ("loss", "sds", "anchor", "vf", "tpc", "sa", "diffusivity"):
            self.assertIn(key, stats)

    def test_optimize_slice_rejects_non_finite_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "latent.*finite"):
            optimize_slice(
                torch.zeros(4, 4, 4),
                NonFiniteEncodeVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                axis=0,
                index=0,
                steps=1,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                sds_weight=0.0,
            )

    def test_optimize_slice_rejects_non_finite_final_decode(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            optimize_slice(
                torch.zeros(4, 4, 4),
                NonFiniteFinalDecodeVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                axis=0,
                index=0,
                steps=1,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                sds_weight=0.0,
            )


class PredictSDSOptimizeVolumeTest(unittest.TestCase):
    def test_optimize_volume_rejects_empty_volume_axes(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            optimize_volume(
                torch.empty(0, 4, 4),
                IdentityVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                steps=1,
                slice_steps=0,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
            )

    def test_optimize_volume_validates_optimizer_contract_when_steps_zero(self):
        volume = torch.zeros(4, 4, 4)
        common = dict(
            volume=volume,
            vae=IdentityVAE(),
            diffusion_model=ZeroNoiseModel(),
            ddpm=DDPM(timesteps=4),
            steps=0,
            slice_steps=0,
            t_min=1,
            t_max=3,
            num_phases=2,
        )

        with self.assertRaisesRegex(ValueError, "lr"):
            optimize_volume(**common, lr=-1.0)

        with self.assertRaisesRegex(ValueError, "sds_weight"):
            optimize_volume(**common, lr=0.1, sds_weight=-1.0)

        with self.assertRaisesRegex(ValueError, "vf_targets"):
            optimize_volume(**common, lr=0.1, vf_weight=1.0)

    def test_optimize_volume_runs_scheduled_slices(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPM(timesteps=4)

        updated, stats = optimize_volume(
            volume,
            vae,
            model,
            ddpm,
            steps=2,
            slice_steps=2,
            lr=0.5,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 0), (1, 1)],
            sds_weight=0.0,
            vf_targets={0: 1.0, 1: 0.0},
            vf_weight=1.0,
            temperature=0.5,
        )

        self.assertEqual(updated.shape, volume.shape)
        self.assertLess(float(updated[0].mean()), 0.0)
        self.assertLess(float(updated[:, 1, :].mean()), 0.0)
        self.assertIn("loss", stats)
        self.assertEqual(int(stats["steps"]), 2)

    def test_optimize_volume_batches_same_axis_slices_for_sds_prior(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = RecordingNoiseModel()
        ddpm = DDPM(timesteps=4)

        updated, stats = optimize_volume(
            volume,
            vae,
            model,
            ddpm,
            steps=1,
            slice_steps=1,
            sds_batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 1), (0, 3)],
            sds_weight=1.0,
        )

        self.assertEqual(model.batch_sizes, [2])
        self.assertEqual(updated.shape, volume.shape)
        self.assertIn("sds", stats)

    def test_objective_batch_descriptor_loss_averages_per_slice_losses(self):
        decoded = torch.stack(
            [
                torch.full((4, 4), 0.0),
                torch.full((4, 4), 1.0),
            ]
        )
        latent = decoded.view(2, 1, 4, 4).clone()

        total, stats = _objective_batch(
            latent,
            decoded,
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_targets=[None, None],
            anchor_weight=0.0,
            vf_targets=torch.tensor([0.5, 0.5]),
            vf_weight=1.0,
            tpc_targets=None,
            tpc_weight=0.0,
            sa_targets=None,
            sa_weight=0.0,
            diffusivity_targets=None,
            diffusivity_solver=None,
            diffusivity_weight=0.0,
            temperature=0.01,
            sa_kernel_size=7,
            sa_sigma=1.0,
        )

        self.assertGreater(float(total.detach()), 0.1)
        self.assertGreater(float(stats["vf"]), 0.1)

    def test_objective_batch_anchor_loss_includes_unanchored_slices_in_mean(self):
        decoded = torch.zeros(2, 4, 4)
        latent = decoded.view(2, 1, 4, 4).clone()
        target = torch.ones(4, 4)

        single_total, _ = _objective(
            latent[:1],
            decoded[0],
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_target=target,
            anchor_weight=1.0,
            vf_targets=None,
            vf_weight=0.0,
            tpc_targets=None,
            tpc_weight=0.0,
            sa_targets=None,
            sa_weight=0.0,
            diffusivity_targets=None,
            diffusivity_solver=None,
            diffusivity_weight=0.0,
            temperature=0.5,
            sa_kernel_size=7,
            sa_sigma=1.0,
        )
        batch_total, stats = _objective_batch(
            latent,
            decoded,
            ZeroNoiseModel(),
            DDPM(timesteps=4),
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            anchor_targets=[target, None],
            anchor_weight=1.0,
            vf_targets=None,
            vf_weight=0.0,
            tpc_targets=None,
            tpc_weight=0.0,
            sa_targets=None,
            sa_weight=0.0,
            diffusivity_targets=None,
            diffusivity_solver=None,
            diffusivity_weight=0.0,
            temperature=0.5,
            sa_kernel_size=7,
            sa_sigma=1.0,
        )

        self.assertTrue(torch.allclose(batch_total, single_total / 2.0))
        self.assertTrue(torch.allclose(stats["anchor"], single_total.detach() / 2.0))

    def test_optimize_volume_rejects_cross_axis_slice_batch(self):
        with self.assertRaisesRegex(ValueError, "same axis"):
            optimize_volume(
                torch.zeros(4, 4, 4),
                IdentityVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                steps=1,
                slice_steps=1,
                sds_batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                slice_schedule=[(0, 1), (1, 1)],
            )

    def test_optimize_volume_uses_matching_anchor_as_soft_loss_only(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPM(timesteps=4)
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=2,
        )

        updated, stats = optimize_volume(
            volume,
            vae,
            model,
            ddpm,
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
        )

        self.assertGreater(float(updated[2].mean()), 0.0)
        self.assertLess(float(updated[2].mean()), 1.0)
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertIn("anchor", stats)

    def test_optimize_volume_rejects_non_finite_batched_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "latent.*finite"):
            optimize_volume(
                torch.zeros(4, 4, 4),
                NonFiniteEncodeVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                steps=1,
                slice_steps=1,
                sds_batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                slice_schedule=[(0, 0), (0, 1)],
                sds_weight=0.0,
            )

    def test_optimize_volume_rejects_non_finite_batched_final_decode(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            optimize_volume(
                torch.zeros(4, 4, 4),
                NonFiniteFinalDecodeVAE(),
                ZeroNoiseModel(),
                DDPM(timesteps=4),
                steps=1,
                slice_steps=1,
                sds_batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                slice_schedule=[(0, 0), (0, 1)],
                sds_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
