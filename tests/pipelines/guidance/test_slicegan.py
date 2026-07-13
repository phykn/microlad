import unittest

import torch
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.conditioning.validation import (
    validate_anchor_intersections,
)
from src.pipelines.guidance.diagnostics import phase_volume_diagnostics
from src.pipelines.guidance.slices import (
    critic_slices,
    transition_profile,
    volume_slices,
)
from src.pipelines.guidance.slicegan import (
    _candidate_steps,
    _training_probabilities,
    anchor_volume_patch,
    anchor_preservation_weights,
    build_anchor_references,
    build_diffusion_references,
    multiscale_shape_loss,
    noise_distribution_loss,
    volume_slice,
)


class FakeSampler:
    def sample(self, shape):
        return torch.zeros(shape)


class CategoricalVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 64
        self.latent_size = 2
        self.latent_ch = 1
        self.num_phases = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        probabilities = torch.full(
            (latent.shape[0], 2, 64, 64),
            0.5,
            device=latent.device,
        )
        return probabilities


class SliceGANTest(unittest.TestCase):
    def test_large_training_forward_checkpoint_preserves_output_and_gradients(self):
        class TinyGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.7))

            def forward(self, noise):
                return torch.sin(noise * self.scale)

        direct_generator = TinyGenerator()
        checkpointed_generator = TinyGenerator()
        checkpointed_generator.load_state_dict(direct_generator.state_dict())
        direct_noise = torch.randn(1, 1, 9, 2, 2, requires_grad=True)
        checkpointed_noise = direct_noise.detach().clone().requires_grad_(True)

        direct = direct_generator(direct_noise)
        checkpointed = _training_probabilities(
            checkpointed_generator,
            checkpointed_noise,
        )
        direct.square().mean().backward()
        checkpointed.square().mean().backward()

        self.assertTrue(torch.equal(checkpointed, direct))
        self.assertTrue(
            torch.allclose(checkpointed_noise.grad, direct_noise.grad, atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(
                checkpointed_generator.scale.grad,
                direct_generator.scale.grad,
                atol=1e-7,
            )
        )

    def test_only_fully_trained_primary_checkpoint_is_conditioned(self):
        self.assertEqual(_candidate_steps(5000), (5000,))

    def test_anchor_references_preserve_labels_and_all_dihedral_views(self):
        anchor = torch.zeros(64, 64, dtype=torch.long)
        anchor[:16] = 1

        references = build_anchor_references(anchor, num_phases=2)

        self.assertEqual(references.shape, torch.Size([8, 2, 64, 64]))
        self.assertTrue(torch.equal(references.sum(dim=1), torch.ones(8, 64, 64)))
        self.assertTrue(
            torch.allclose(
                references[:, 1].mean(dim=(1, 2)),
                torch.full((8,), 0.25),
            )
        )

    def test_diffusion_references_are_calibrated_to_target_fraction(self):
        references = build_diffusion_references(
            FakeSampler(),
            CategoricalVAE(),
            target_fraction=torch.tensor([0.75, 0.25]),
            num_phases=2,
            count=2,
        )

        self.assertEqual(references.shape, torch.Size([2, 2, 64, 64]))
        self.assertTrue(
            torch.allclose(
                references.mean(dim=(0, 2, 3)),
                torch.tensor([0.75, 0.25]),
            )
        )

    def test_volume_slices_preserve_every_plane_for_each_axis(self):
        labels = torch.zeros(64, 64, 64, dtype=torch.long)
        labels[1] = 1
        labels[:, 2] = 1
        probabilities = (
            torch.nn.functional.one_hot(
                labels,
                num_classes=2,
            )
            .movedim(-1, 0)
            .unsqueeze(0)
            .float()
        )

        axis_zero = volume_slices(probabilities, 0, num_phases=2)
        axis_one = volume_slices(probabilities, 1, num_phases=2)
        axis_two = volume_slices(probabilities, 2, num_phases=2)

        self.assertEqual(axis_zero.shape, torch.Size([64, 2, 64, 64]))
        self.assertTrue(torch.equal(axis_zero[1].argmax(dim=0), labels[1]))
        self.assertTrue(torch.equal(axis_one[2].argmax(dim=0), labels[:, 2, :]))
        self.assertTrue(torch.equal(axis_two[3].argmax(dim=0), labels[:, :, 3]))

    def test_volume_slice_reads_labels_and_probabilities_on_every_axis(self):
        labels = torch.arange(4 * 4 * 4).reshape(4, 4, 4)
        probabilities = torch.stack([labels, labels + 100])

        self.assertTrue(torch.equal(volume_slice(labels, 0, 1), labels[1]))
        self.assertTrue(torch.equal(volume_slice(labels, 1, 2), labels[:, 2]))
        self.assertTrue(torch.equal(volume_slice(labels, 2, 3), labels[:, :, 3]))
        self.assertTrue(
            torch.equal(volume_slice(probabilities, 1, 2), probabilities[:, :, 2])
        )

    def test_scale_anchor_patch_uses_absolute_plane_and_centered_in_plane_area(self):
        volume = torch.zeros(128, 128, 128, dtype=torch.uint8)
        volume[32:96, 32:96, 84] = 1
        anchor = VolumeAnchor(
            image=torch.ones(64, 64),
            axis=2,
            index=84,
            start=32,
        )

        patch = anchor_volume_patch(volume, anchor)

        self.assertEqual(patch.shape, torch.Size([64, 64]))
        self.assertTrue(torch.equal(patch, torch.ones(64, 64, dtype=torch.uint8)))

    def test_anchor_intersections_accept_compatible_cross_axis_lines(self):
        image = torch.zeros(64, 64)
        anchors = [
            VolumeAnchor(image=image, axis=0, index=32),
            VolumeAnchor(image=image.clone(), axis=1, index=32),
        ]

        validate_anchor_intersections(anchors, tolerance=0.0)

    def test_anchor_intersections_reject_conflicting_cross_axis_lines(self):
        anchors = [
            VolumeAnchor(image=torch.zeros(64, 64), axis=0, index=32),
            VolumeAnchor(image=torch.ones(64, 64), axis=1, index=32),
        ]

        with self.assertRaisesRegex(ValueError, "Conflicting anchor intersection"):
            validate_anchor_intersections(anchors, tolerance=0.1)

    def test_anchor_intersections_reject_duplicate_plane(self):
        image = torch.zeros(64, 64)
        anchors = [
            VolumeAnchor(image=image, axis=2, index=12),
            VolumeAnchor(image=image.clone(), axis=2, index=12),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate anchor slice"):
            validate_anchor_intersections(anchors, tolerance=0.1)

    def test_centered_scale_anchor_intersections_use_absolute_indices(self):
        image = torch.zeros(64, 64)
        image[32, :] = 1
        anchors = [
            VolumeAnchor(
                image=image,
                axis=0,
                index=64,
                start=32,
            ),
            VolumeAnchor(
                image=image.clone(),
                axis=1,
                index=64,
                start=32,
            ),
        ]

        validate_anchor_intersections(anchors, tolerance=0.0)

    def test_offset_scale_anchor_intersections_use_patch_local_rows(self):
        xy = torch.zeros(64, 64)
        xz = torch.zeros(64, 64)
        pattern = torch.arange(64) % 2
        xy[48, :] = pattern
        xz[8, :] = pattern
        anchors = [
            VolumeAnchor(image=xy, axis=0, index=40, start=32),
            VolumeAnchor(image=xz, axis=1, index=80, start=32),
        ]

        validate_anchor_intersections(anchors, tolerance=0.0)
        anchors[1].image[8, :] = 1 - pattern
        with self.assertRaisesRegex(ValueError, "Conflicting anchor intersection"):
            validate_anchor_intersections(anchors, tolerance=0.0)

    def test_non_overlapping_anchor_patches_do_not_create_false_conflicts(self):
        anchors = [
            VolumeAnchor(
                image=torch.zeros(64, 64),
                axis=0,
                index=0,
                start=0,
            ),
            VolumeAnchor(
                image=torch.ones(64, 64),
                axis=1,
                index=127,
                start=64,
            ),
        ]

        validate_anchor_intersections(anchors, tolerance=0.0)

    def test_large_volume_critic_uses_bounded_64_pixel_patches(self):
        volume = torch.randn(1, 2, 96, 96, 96)

        slices = critic_slices(volume, 1, num_phases=2)

        self.assertEqual(slices.shape, torch.Size([64, 2, 64, 64]))

    def test_categorical_diagnostics_detect_repetition_and_global_cutoff(self):
        labels = torch.zeros(16, 16, 16, dtype=torch.long)
        labels[:, :, 1::2] = 1
        references = (
            torch.nn.functional.one_hot(labels[0], num_classes=2)
            .movedim(-1, 0)
            .unsqueeze(0)
            .float()
        )

        diagnostics = phase_volume_diagnostics(
            labels,
            references,
            target_fraction=torch.tensor([0.5, 0.5]),
            num_phases=2,
            run_lengths=(2, 4, 8),
        )

        self.assertEqual(
            diagnostics["axis_run_profile"].shape,
            torch.Size([3, 2, 3]),
        )
        self.assertEqual(
            diagnostics["axis_euler_density"].shape,
            torch.Size([3, 2]),
        )
        self.assertAlmostEqual(
            float(diagnostics["axis_exact_repeat_rate"][0]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["axis_exact_repeat_rate"][2]),
            0.0,
        )
        self.assertTrue(
            torch.allclose(
                diagnostics["phase_fraction_error"],
                torch.zeros(2),
            )
        )

        cutoff = torch.zeros(16, 16, 16, dtype=torch.long)
        cutoff[8:] = 1
        cutoff_diagnostics = phase_volume_diagnostics(
            cutoff,
            references,
            target_fraction=torch.tensor([0.5, 0.5]),
            num_phases=2,
            run_lengths=(2, 4, 8),
        )
        self.assertAlmostEqual(
            float(cutoff_diagnostics["axis_global_boundary_jump"][0]),
            1.0,
        )

    def test_shape_and_noise_losses_have_expected_minima(self):
        target = torch.zeros(2, 64, 64)
        target[0, :, :32] = 1.0
        target[1, :, 32:] = 1.0
        changed = target.roll(8, dims=-1)

        self.assertEqual(float(multiscale_shape_loss(target, target)), 0.0)
        self.assertGreater(float(multiscale_shape_loss(changed, target)), 0.0)
        self.assertTrue(
            torch.isfinite(noise_distribution_loss(torch.randn(1, 32, 4, 4, 4)))
        )

    def test_transition_profile_detects_changes_without_smoothing(self):
        labels = torch.zeros(4, 4, 4, dtype=torch.long)
        labels[1::2] = 1
        probabilities = (
            torch.nn.functional.one_hot(
                labels,
                num_classes=2,
            )
            .movedim(-1, 0)
            .float()
        )

        axis_zero = transition_profile(probabilities, 0)
        axis_one = transition_profile(probabilities, 1)

        self.assertTrue(torch.equal(axis_zero, torch.ones(3)))
        self.assertTrue(torch.equal(axis_one, torch.zeros(3)))

    def test_anchor_preservation_weights_allow_local_change_without_hard_slab(self):
        anchors = [
            VolumeAnchor(torch.zeros(9, 9), axis=0, index=4),
            VolumeAnchor(torch.zeros(9, 9), axis=1, index=4),
        ]

        weights = anchor_preservation_weights(
            (9, 9, 9),
            anchors,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sigma=2.0,
        )[0, 0]

        self.assertTrue(torch.equal(weights[4], torch.zeros(9, 9)))
        self.assertTrue(torch.equal(weights[:, 4], torch.zeros(9, 9)))
        self.assertGreater(float(weights[8, 0, 0]), float(weights[5, 0, 0]))


if __name__ == "__main__":
    unittest.main()
