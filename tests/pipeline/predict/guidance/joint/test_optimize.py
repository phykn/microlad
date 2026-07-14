import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.pipeline.predict.guidance.latent_slices import sample_slices as sample_latent_slices
from src.modeling.diffusion import DDPMProcess
from src.pipeline.predict.guidance.joint.loss import (
    anchor_loss,
    axis_loss,
    axis_mass_loss,
    fraction_loss,
)
from src.pipeline.predict.guidance.joint.model import LatentRefiner
from src.pipeline.predict.guidance.joint.optimize import optimize_latent
from src.pipeline.predict.guidance.joint.slices import (
    extract_slices,
    phase_values,
    select_slices,
)
from src.pipeline.predict.reconstruction.volume import decode_axis_probs, decode_volume_probs


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


class RecordingCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(latent.shape[0]))
        return latent.mean(dim=(1, 2, 3), keepdim=True)


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
    def test_axis_loss_measures_decoded_disagreement(self):
        probabilities = torch.tensor(
            [
                [[[[0.9, 0.1]]], [[[0.1, 0.9]]]],
                [[[[0.1, 0.9]]], [[[0.9, 0.1]]]],
                [[[[0.5, 0.5]]], [[[0.5, 0.5]]]],
            ]
        )

        disagreement = axis_loss(probabilities)
        mean = probabilities.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1, 1)

        self.assertGreater(float(disagreement), 0.0)
        self.assertEqual(float(axis_loss(mean)), 0.0)

    def test_axis_loss_does_not_reward_removing_a_phase(self):
        probabilities = torch.tensor(
            [
                [[[[0.8, 0.8]]], [[[0.2, 0.2]]]],
                [[[[0.4, 0.4]]], [[[0.6, 0.6]]]],
                [[[[0.6, 0.6]]], [[[0.4, 0.4]]]],
            ]
        )

        self.assertAlmostEqual(float(axis_loss(probabilities)), 0.0, places=6)

    def test_axis_mass_loss_detects_phase_fraction_disagreement(self):
        probabilities = torch.tensor(
            [
                [[[[0.8, 0.8]]], [[[0.2, 0.2]]]],
                [[[[0.4, 0.4]]], [[[0.6, 0.6]]]],
                [[[[0.6, 0.6]]], [[[0.4, 0.4]]]],
            ]
        )
        matched = probabilities.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1, 1)

        self.assertGreater(float(axis_mass_loss(probabilities)), 0.0)
        self.assertEqual(float(axis_mass_loss(matched)), 0.0)

    def test_joint_records_axis_mass_consistency(self):
        _, history = optimize_latent(
            torch.zeros(1, 2, 2, 2),
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            batch_size=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            axis_weight=0.0,
            axis_mass_weight=0.25,
            continuity_weight=0.0,
            preservation_weight=0.0,
        )

        self.assertIn("axis_mass", history)
        self.assertEqual(history["axis_mass"].shape, torch.Size([1]))

    def test_anchor_loss_balances_present_phases(self):
        probabilities = torch.tensor(
            [
                [[[0.9, 0.9, 0.9, 0.9]]],
                [[[0.1, 0.1, 0.1, 0.1]]],
            ]
        )
        target = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])

        loss = anchor_loss(
            probabilities,
            target,
            torch.ones_like(target),
        )
        expected = -0.5 * (torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.1)))

        self.assertTrue(torch.allclose(loss, expected))

    def test_fraction_loss_is_zero_at_target_and_preserves_absent_phase_gradient(self):
        target = torch.tensor([0.5, 0.5])
        matched = target.clone().requires_grad_()
        missing = torch.tensor([1.0, 0.0], requires_grad=True)

        self.assertAlmostEqual(
            float(fraction_loss(matched, target).detach()),
            0.0,
            places=6,
        )
        loss = fraction_loss(missing, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(missing.grad).all())
        self.assertNotEqual(float(missing.grad[1]), 0.0)

    def test_zero_initialized_refiner_preserves_lmpdd_latent(self):
        latent = torch.randn(1, 2, 2, 2).unsqueeze(0)
        refiner = LatentRefiner(1, scale=0.25)

        refined = refiner(latent)

        self.assertTrue(torch.equal(refined, latent))

    def test_refiner_residual_uses_each_channel_standard_deviation(self):
        base = torch.tensor(
            [
                [
                    [[[0.0, 2.0], [0.0, 2.0]]],
                    [[[0.0, 6.0], [0.0, 6.0]]],
                ]
            ]
        )
        refiner = LatentRefiner(2, scale=0.5)
        with torch.no_grad():
            refiner.to_residual.bias.fill_(1.0)

        delta = refiner(base) - base
        ratio = delta[:, 1].mean() / delta[:, 0].mean()

        self.assertTrue(torch.allclose(ratio, torch.tensor(3.0)))

    def test_zero_steps_returns_only_original_latent(self):
        latent = torch.randn(1, 2, 2, 2)

        refined, history = optimize_latent(
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

        self.assertTrue(torch.equal(refined, latent))
        self.assertEqual(history["step"].numel(), 0)

    def test_pretrained_critic_guides_shared_latent_without_condition(self):
        critic = RecordingCritic()
        _, stats = optimize_latent(
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
            sds_weight=0.0,
            critic=critic,
            critic_weight=0.1,
            axis_weight=0.0,
            continuity_weight=0.0,
            preservation_weight=0.0,
        )

        self.assertIn("critic", stats)
        self.assertEqual(critic.batch_sizes, [2])

    def test_unlimited_decode_batch_disables_checkpointing(self):
        latent = torch.randn(1, 2, 2, 2)

        with patch(
            "src.pipeline.predict.guidance.joint.optimize.decode_axis_probs",
            wraps=decode_axis_probs,
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
            "src.pipeline.predict.guidance.joint.optimize.tqdm",
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
                anchor_weight=5.0,
                fraction_targets=torch.tensor([0.5, 0.5]),
                global_fraction_weight=1.0,
                progress=True,
            )

        progress = FakeProgress.instances[0]
        self.assertEqual(progress.kwargs["desc"], "Joint guidance")
        self.assertEqual(
            set(progress.postfixes[-1]),
            {"loss", "anchor", "fraction", "axis"},
        )
        self.assertAlmostEqual(
            float(progress.postfixes[-1]["anchor"]),
            float(torch.log(torch.tensor(2.0))),
            places=3,
        )

    def test_global_fraction_uses_hard_forward_values(self):
        _, stats = optimize_latent(
            torch.zeros(1, 2, 2, 2),
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            batch_size=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            fraction_targets=torch.tensor([0.0, 1.0]),
            global_fraction_weight=1.0,
            sds_weight=0.0,
            continuity_weight=0.0,
            preservation_weight=0.0,
        )

        self.assertGreater(float(stats["global_fraction"][-1]), 10.0)

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

        refined, history = optimize_latent(
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
        )

        after = decode_volume_probs(vae, refined)[0, 1, 1].mean()
        self.assertGreater(float(after), float(before))
        self.assertIn("anchor", history)
        self.assertEqual(history["step"].tolist(), list(range(1, 13)))
        self.assertEqual(history["loss"].shape, torch.Size([12]))

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

    def test_single_slice_prior_cycles_axes_across_steps(self):
        with patch(
            "src.pipeline.predict.guidance.joint.optimize.sample_slices",
            wraps=sample_latent_slices,
        ) as sampler:
            optimize_latent(
                torch.zeros(1, 2, 2, 2),
                IdentityCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=6,
                batch_size=1,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                continuity_weight=0.0,
                preservation_weight=0.0,
            )

        self.assertEqual(
            [call.kwargs["axis_offset"] for call in sampler.call_args_list],
            [0, 1, 2, 0, 1, 2],
        )

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
