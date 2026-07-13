import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.modeling.diffusion import DDPMProcess
from src.pipelines.guidance.joint.model import LatentRefiner
from src.pipelines.guidance.joint.optimize import optimize_latent
from src.pipelines.guidance.joint.slices import (
    extract_slices,
    phase_values,
    select_slices,
)
from src.pipelines.reconstruction.volume import decode_volume_probs


class IdentityCategoricalVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2
    downsample_factor = 1

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.cat([-latent, latent], dim=1), dim=1)


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent)


class FakeProgress:
    instances = []

    def __init__(self, iterable, **kwargs) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.postfixes = []
        self.__class__.instances.append(self)

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values) -> None:
        self.postfixes.append(values)


class JointOptimizationTest(unittest.TestCase):
    def test_zero_initialized_refiner_preserves_lmpdd_latent(self):
        latent = torch.randn(1, 2, 2, 2).unsqueeze(0)
        refiner = LatentRefiner(1, scale=0.25)

        refined = refiner(latent)

        self.assertTrue(torch.equal(refined, latent))

    def test_zero_steps_returns_only_original_latent(self):
        latent = torch.randn(1, 2, 2, 2)

        candidates, stats = optimize_latent(
            latent,
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=0,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(torch.equal(candidates[0], latent))
        self.assertEqual(stats["joint_candidate_steps"].tolist(), [0])

    def test_unlimited_decode_batch_disables_checkpointing(self):
        latent = torch.randn(1, 2, 2, 2)

        with patch(
            "src.pipelines.guidance.joint.optimize.decode_volume_probs",
            wraps=decode_volume_probs,
        ) as decode:
            optimize_latent(
                latent,
                IdentityCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                batch_size=2,
                decode_batch_size=None,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
            )

        self.assertIsNone(decode.call_args.kwargs["plane_batch_size"])
        self.assertFalse(decode.call_args.kwargs["checkpoint_gradients"])

    def test_joint_progress_shows_live_condition_losses(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        FakeProgress.instances = []

        with patch(
            "src.pipelines.guidance.joint.optimize.tqdm",
            FakeProgress,
        ):
            optimize_latent(
                torch.zeros(1, 2, 2, 2),
                IdentityCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                anchors=[anchor],
                anchor_weight=1.0,
                vf_targets=torch.tensor([0.5, 0.5]),
                vf_weight=1.0,
                progress=True,
            )

        progress = FakeProgress.instances[0]
        self.assertEqual(progress.kwargs["desc"], "Joint guidance")
        self.assertEqual(
            set(progress.postfixes[-1]),
            {"loss", "anchor", "vf"},
        )

    def test_joint_anchor_loss_changes_decoded_anchor_without_copying(self):
        torch.manual_seed(0)
        vae = IdentityCategoricalVAE()
        latent = torch.zeros(1, 2, 2, 2)
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        before = decode_volume_probs(vae, latent)[0, 1, 1].mean()

        candidates, stats = optimize_latent(
            latent,
            vae,
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=12,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
            continuity_weight=0.0,
            preservation_weight=0.0,
            checkpoint_every=6,
        )

        after = decode_volume_probs(vae, candidates[-1])[0, 1, 1].mean()
        self.assertGreater(float(after), float(before))
        self.assertIn("history_anchor", stats)
        self.assertEqual(stats["joint_candidate_steps"].tolist(), [0, 6, 12])

    def test_invalid_latent_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "joint latent"):
            optimize_latent(
                torch.zeros(1, 2, 2),
                IdentityCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=0,
                batch_size=1,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
            )

    def test_joint_slice_sampler_balances_axes(self):
        axes = [
            select_slices(
                step,
                size=4,
                batch_size=2,
                device=torch.device("cpu"),
            )[0]
            for step in range(18)
        ]

        self.assertEqual([axes.count(axis) for axis in range(3)], [6, 6, 6])

    def test_probability_slice_extraction_preserves_axis_coordinates(self):
        values = torch.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3).float()

        xy = extract_slices(values, axis=0, indices=[1])
        xz = extract_slices(values, axis=1, indices=[1])
        yz = extract_slices(values, axis=2, indices=[1])

        self.assertTrue(torch.equal(xy[0], values[:, 1]))
        self.assertTrue(torch.equal(xz[0], values[:, :, 1, :]))
        self.assertTrue(torch.equal(yz[0], values[:, :, :, 1]))

    def test_straight_through_values_keep_soft_gradients(self):
        probabilities = torch.tensor([[[[0.6]], [[0.4]]]], requires_grad=True)

        values = phase_values(probabilities, num_phases=2)
        values.sum().backward()

        self.assertEqual(float(values.item()), 0.0)
        self.assertGreater(float(probabilities.grad.abs().sum()), 0.0)

if __name__ == "__main__":
    unittest.main()
