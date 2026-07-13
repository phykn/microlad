import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.modeling.diffusion import DDPMProcess
from src.app.api import AnchorSlice
from src.pipelines.guidance.sds.optimize import optimize_volume
from src.pipelines.guidance.sds.slice import optimize_slice
from src.pipelines.guidance.sds.optimize import _smooth_anchor_slabs
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.sds.loss import batch_loss, slice_loss


class IdentityVAE(nn.Module):
    image_size = 4
    latent_size = 4
    latent_ch = 1
    num_phases = 2

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.clone(), torch.zeros_like(x)

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        base = torch.where(z >= 0.0, torch.tanh(z), torch.zeros_like(z))
        phase_one = 1e-3 + (1.0 - 2e-3) * base
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class NonFiniteEncodeVAE(IdentityVAE):
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(x, float("nan")), torch.zeros_like(x)

class NonFiniteFinalDecodeVAE(IdentityVAE):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1

        if self.decode_calls > 1:
            return torch.full(
                (z.shape[0], self.num_phases, *z.shape[-2:]),
                float("nan"),
                dtype=z.dtype,
                device=z.device,
            )

        return super().decode_probs(z)


class CategoricalVAE(IdentityVAE):
    num_phases = 2

    def __init__(self) -> None:
        super().__init__()
        self.probability_calls = 0

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        self.probability_calls += 1
        return torch.softmax(torch.cat([-z, z], dim=1), dim=1)


class ThreePhaseCategoricalVAE(IdentityVAE):
    num_phases = 3

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.cat([-z, torch.zeros_like(z), z], dim=1), dim=1)


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
    def test_optimize_slice_uses_categorical_decoder_probabilities(self):
        vae = CategoricalVAE()
        volume = torch.zeros(4, 4, 4)

        updated, stats = optimize_slice(
            volume,
            vae,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            axis=0,
            index=1,
            steps=2,
            lr=0.5,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            vf_targets={0: 1.0, 1: 0.0},
            vf_weight=1.0,
        )

        self.assertGreaterEqual(vae.probability_calls, 3)
        self.assertTrue(torch.all(updated[1] == updated[1].round()))
        self.assertLess(float(updated[1].mean()), 0.5)
        self.assertIn("vf", stats)

    def test_optimize_slice_updates_only_selected_slice(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPMProcess(timesteps=4)

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
            vf_targets={0: 0.0, 1: 1.0},
            vf_weight=1.0,
            temperature=0.5,
        )

        self.assertTrue(torch.allclose(volume, torch.zeros_like(volume)))
        self.assertGreater(float(updated[1].mean()), 0.0)
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertTrue(torch.allclose(updated[2], volume[2]))
        self.assertTrue(torch.allclose(updated[3], volume[3]))
        self.assertIn("loss", stats)
        self.assertIn("vf", stats)

    def test_optimize_slice_uses_soft_anchor_loss_without_forced_overwrite(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPMProcess(timesteps=4)
        anchor_target = torch.ones(1, 1, 4, 4)

        updated, stats, probabilities = optimize_slice(
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
            return_probabilities=True,
        )

        self.assertTrue(torch.all(updated[:, 2, :] == 0.0))
        self.assertGreater(float(probabilities[:, 1].mean()), 0.0)
        self.assertLess(float(probabilities[:, 1].mean()), 1.0)
        self.assertIn("anchor", stats)
        self.assertTrue(torch.allclose(updated[:, 0, :], volume[:, 0, :]))

    def test_optimize_slice_combines_all_sds_objectives(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = ZeroNoiseModel()
        ddpm = DDPMProcess(timesteps=4)
        diffusivity_solver = ConductanceSolver(height=2, width=2, low_cond=0.1)

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
                DDPMProcess(timesteps=4),
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
                DDPMProcess(timesteps=4),
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
                DDPMProcess(timesteps=4),
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
            ddpm=DDPMProcess(timesteps=4),
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
        ddpm = DDPMProcess(timesteps=4)

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
            vf_targets={0: 0.0, 1: 1.0},
            vf_weight=1.0,
            temperature=0.5,
        )

        self.assertEqual(updated.shape, volume.shape)
        self.assertGreater(float(updated[0].mean()), 0.0)
        self.assertGreater(float(updated[:, 1, :].mean()), 0.0)
        self.assertIn("history_loss", stats)
        self.assertEqual(int(stats["steps"]), 2)

    def test_optimize_volume_batches_same_axis_slices_for_sds_prior(self):
        volume = torch.zeros(4, 4, 4)
        vae = IdentityVAE()
        model = RecordingNoiseModel()
        ddpm = DDPMProcess(timesteps=4)

        updated, stats = optimize_volume(
            volume,
            vae,
            model,
            ddpm,
            steps=1,
            slice_steps=1,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            slice_schedule=[(0, 1), (0, 3)],
            sds_weight=1.0,
        )

        self.assertEqual(model.batch_sizes, [2])
        self.assertEqual(updated.shape, volume.shape)
        self.assertIn("history_sds", stats)

    def test_objective_batch_descriptor_loss_averages_per_slice_losses(self):
        decoded = torch.stack(
            [
                torch.full((4, 4), 0.0),
                torch.full((4, 4), 1.0),
            ]
        )
        latent = decoded.view(2, 1, 4, 4).clone()

        total, stats = batch_loss(
            latent,
            decoded,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
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

    def test_objective_batch_anchor_loss_is_not_diluted_by_unanchored_slices(self):
        decoded = torch.zeros(2, 4, 4)
        latent = decoded.view(2, 1, 4, 4).clone()
        target = torch.ones(4, 4)

        single_total, _ = slice_loss(
            latent[:1],
            decoded[0],
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
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
        batch_total, stats = batch_loss(
            latent,
            decoded,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
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

        self.assertTrue(torch.allclose(batch_total, single_total))
        self.assertTrue(torch.allclose(stats["anchor"], single_total.detach()))

    def test_optimize_volume_rejects_cross_axis_slice_batch(self):
        with self.assertRaisesRegex(ValueError, "same axis"):
            optimize_volume(
                torch.zeros(4, 4, 4),
                IdentityVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                slice_steps=1,
                batch_size=2,
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
        ddpm = DDPMProcess(timesteps=4)
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

        self.assertTrue(torch.all(updated[2] == updated[2].round()))
        self.assertFalse(torch.equal(updated[2], torch.ones_like(updated[2])))
        self.assertTrue(torch.allclose(updated[0], volume[0]))
        self.assertIn("history_anchor", stats)

    def test_optimize_volume_applies_anchor_to_cross_axis_intersection(self):
        volume = torch.zeros(4, 4, 4)
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=2,
        )

        updated, stats = optimize_volume(
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
            slice_schedule=[(1, 1)],
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
        )

        self.assertTrue(torch.all(updated[2, 1] == updated[2, 1].round()))
        self.assertTrue(torch.allclose(updated[0, 1], volume[0, 1]))
        self.assertIn("history_anchor", stats)

    def test_consensus_sweep_fuses_three_axis_probabilities(self):
        volume = torch.zeros(4, 4, 4)
        schedule = [
            *[(0, index) for index in range(4)],
            *[(1, index) for index in range(4)],
            *[(2, index) for index in range(4)],
        ]

        def proposal(current, *args, axis, indices, **kwargs):
            probabilities = torch.zeros(len(indices), 3, 4, 4)
            probabilities[:, axis] = 1.0
            return current.clone(), {}, probabilities

        with patch(
            "src.pipelines.guidance.sds.slice.optimize_slice_batch",
            side_effect=proposal,
        ):
            updated, _ = optimize_volume(
                volume,
                ThreePhaseCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=3,
                slice_steps=1,
                batch_size=4,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=3,
                slice_schedule=schedule,
                sds_weight=0.0,
                vf_targets=torch.tensor([0.5, 0.25, 0.25]),
                consensus_sweeps=True,
            )

        counts = torch.bincount(updated.to(torch.long).flatten(), minlength=3)
        self.assertTrue(torch.equal(counts, torch.tensor([32, 16, 16])))

    def test_anchor_slab_smoothing_reduces_center_plane_jump(self):
        probabilities = torch.zeros(2, 5, 2, 2)
        probabilities[0] = 1.0
        probabilities[:, 2] = torch.tensor([0.0, 1.0]).view(2, 1, 1)
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=2,
        )

        before = (probabilities[:, 2] - probabilities[:, 1]).abs().mean()
        smoothed = _smooth_anchor_slabs(
            probabilities,
            [anchor],
            radius=1,
            weight=1.0,
        )
        after = (smoothed[:, 2] - smoothed[:, 1]).abs().mean()

        self.assertLess(float(after), float(before))

    def test_optimize_volume_rejects_non_finite_batched_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "latent.*finite"):
            optimize_volume(
                torch.zeros(4, 4, 4),
                NonFiniteEncodeVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                slice_steps=1,
                batch_size=2,
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
                DDPMProcess(timesteps=4),
                steps=1,
                slice_steps=1,
                batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                slice_schedule=[(0, 0), (0, 1)],
                sds_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
