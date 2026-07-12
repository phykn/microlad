import unittest

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.modeling.diffusion import DDPMProcess
from src.pipelines.guidance.joint_optimization import (
    _axis_transition_loss,
    _build_run_profile_target,
    _build_transition_target,
    _build_interface_target,
    _build_texture_targets,
    _interface_loss,
    _straight_through_phase_values,
    _texture_swd_loss,
    all_axis_probability_slices,
    extract_probability_slices,
    optimize_joint_volume,
    select_joint_slices,
)


class IdentityCategoricalVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2

    def encode(self, image: torch.Tensor):
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.cat([-latent, latent], dim=1), dim=1)


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent)


class PatchCategoricalVAE(IdentityCategoricalVAE):
    image_size = 16
    latent_size = 16


class JointOptimizationTest(unittest.TestCase):
    def test_joint_slice_sampler_balances_axes(self):
        axes = [
            select_joint_slices(
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

        xy = extract_probability_slices(values, axis=0, indices=[1])
        xz = extract_probability_slices(values, axis=1, indices=[1])
        yz = extract_probability_slices(values, axis=2, indices=[1])

        self.assertTrue(torch.equal(xy[0], values[:, 1]))
        self.assertTrue(torch.equal(xz[0], values[:, :, 1, :]))
        self.assertTrue(torch.equal(yz[0], values[:, :, :, 1]))

    def test_all_axis_slices_include_every_plane_from_every_axis(self):
        values = torch.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3).float()

        slices = all_axis_probability_slices(values)

        self.assertEqual(slices.shape, torch.Size([9, 2, 3, 3]))
        for axis in range(3):
            expected = extract_probability_slices(
                values,
                axis=axis,
                indices=[0, 1, 2],
            )
            self.assertTrue(torch.equal(slices[axis * 3 : (axis + 1) * 3], expected))

    def test_straight_through_values_are_categorical_with_soft_gradients(self):
        probabilities = torch.tensor(
            [[[[0.6]], [[0.4]]]],
            requires_grad=True,
        )

        values = _straight_through_phase_values(probabilities, num_phases=2)
        values.sum().backward()

        self.assertEqual(float(values.item()), 0.0)
        self.assertIsNotNone(probabilities.grad)
        self.assertGreater(float(probabilities.grad.abs().sum()), 0.0)

    def test_joint_output_matches_condition_phase_fraction(self):
        initial = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]
        )
        updated, stats = optimize_joint_volume(
            initial,
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=0,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            vf_targets=torch.tensor([0.5, 0.5]),
            entropy_weight=0.0,
            continuity_weight=0.0,
        )

        counts = torch.bincount(updated.to(torch.long).flatten(), minlength=2)
        self.assertTrue(torch.equal(counts, torch.tensor([4, 4])))
        self.assertEqual(int(stats["joint_steps"]), 0)

    def test_joint_anchor_loss_updates_shared_anchor_plane(self):
        torch.manual_seed(0)
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        updated, stats = optimize_joint_volume(
            torch.zeros(2, 2, 2),
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=20,
            batch_size=2,
            lr=0.2,
            t_min=1,
            t_max=3,
            num_phases=2,
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
            entropy_weight=0.0,
            continuity_weight=0.0,
        )

        self.assertTrue(torch.all(updated[1] == 1))
        self.assertIn("anchor", stats)

    def test_joint_anchor_is_not_hard_copied_without_optimization(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        updated, _ = optimize_joint_volume(
            torch.zeros(2, 2, 2),
            IdentityCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=0,
            batch_size=2,
            lr=0.2,
            t_min=1,
            t_max=3,
            num_phases=2,
            anchors=[anchor],
            anchor_weight=1.0,
            sds_weight=0.0,
            entropy_weight=0.0,
            continuity_weight=0.0,
        )

        self.assertTrue(torch.all(updated == 0))

    def test_joint_patch_discriminator_uses_reference_images(self):
        updated, stats = optimize_joint_volume(
            torch.zeros(16, 16, 16),
            PatchCategoricalVAE(),
            ZeroNoiseModel(),
            DDPMProcess(timesteps=4),
            steps=1,
            batch_size=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            num_phases=2,
            sds_weight=0.0,
            entropy_weight=0.0,
            continuity_weight=0.0,
            reference_images=[np.zeros((16, 16), dtype=np.uint8)],
            patch_weight=0.1,
        )

        self.assertEqual(updated.shape, torch.Size([16, 16, 16]))
        self.assertIn("patch", stats)
        self.assertIn("patch_discriminator", stats)

    def test_joint_patch_discriminator_requires_reference_images(self):
        with self.assertRaisesRegex(ValueError, "reference_images"):
            optimize_joint_volume(
                torch.zeros(16, 16, 16),
                PatchCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                sds_weight=0.0,
                reference_images=None,
                patch_weight=0.1,
            )

    def test_texture_swd_is_zero_for_the_same_categorical_image(self):
        torch.manual_seed(0)
        image = np.zeros((16, 16), dtype=np.uint8)
        targets = _build_texture_targets(
            [image],
            num_phases=2,
            image_size=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            enabled=True,
        )
        labels = torch.from_numpy(image).to(torch.long).unsqueeze(0)
        fake = torch.nn.functional.one_hot(labels, num_classes=2).permute(0, 3, 1, 2)

        loss = _texture_swd_loss(fake.float(), targets)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_joint_texture_guidance_requires_reference_images(self):
        with self.assertRaisesRegex(ValueError, "reference_images"):
            optimize_joint_volume(
                torch.zeros(16, 16, 16),
                PatchCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                sds_weight=0.0,
                reference_images=None,
                texture_weight=0.1,
            )

    def test_interface_loss_matches_phase_pair_boundaries_per_slice(self):
        image = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.uint8)
        target = _build_interface_target(
            [image],
            num_phases=2,
            image_size=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            enabled=True,
        )
        labels = torch.from_numpy(image).to(torch.long).unsqueeze(0)
        matching = torch.nn.functional.one_hot(labels, num_classes=2).permute(0, 3, 1, 2)
        uniform = torch.zeros_like(matching)
        uniform[:, 0] = 1

        matching_loss = _interface_loss(matching.float(), target)
        uniform_loss = _interface_loss(uniform.float(), target)

        self.assertAlmostEqual(float(matching_loss), 0.0, places=6)
        self.assertGreater(float(uniform_loss), 0.0)

    def test_axis_transition_loss_matches_reference_boundary_rate(self):
        image = (np.indices((4, 4)).sum(axis=0) % 2).astype(np.uint8)
        target = _build_transition_target(
            [image],
            num_phases=2,
            image_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            enabled=True,
        )
        labels = torch.from_numpy(
            (np.indices((4, 4, 4)).sum(axis=0) % 2).astype(np.int64)
        ).unsqueeze(0)
        probabilities = torch.nn.functional.one_hot(
            labels,
            num_classes=2,
        ).movedim(-1, 1).float()

        loss, rates = _axis_transition_loss(probabilities, target)

        self.assertAlmostEqual(float(target), 1.0, places=6)
        self.assertTrue(torch.allclose(rates, torch.ones(3)))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_run_profile_target_averages_both_image_directions(self):
        image = (np.indices((16, 16))[0] % 2).astype(np.uint8)

        target = _build_run_profile_target(
            [image],
            num_phases=2,
            image_size=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            lengths=(2, 4, 8, 16),
            enabled=True,
        )

        self.assertEqual(target.shape, torch.Size([2, 4]))
        self.assertTrue(torch.all(target >= 0.0))
        self.assertTrue(torch.all(target <= 1.0))

    def test_joint_run_profile_requires_reference_images(self):
        with self.assertRaisesRegex(ValueError, "reference_images"):
            optimize_joint_volume(
                torch.zeros(16, 16, 16),
                PatchCategoricalVAE(),
                ZeroNoiseModel(),
                DDPMProcess(timesteps=4),
                steps=1,
                batch_size=2,
                lr=0.1,
                t_min=1,
                t_max=3,
                num_phases=2,
                sds_weight=0.0,
                reference_images=None,
                run_weight=0.1,
            )



if __name__ == "__main__":
    unittest.main()
