import unittest

import torch
import torch.nn.functional as F

from src.app.api.options import SliceGANConditionConfig
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume
from src.modeling.slicegan.sampling import sample_slices
from src.pipelines.guidance.slicegan.anchors import (
    PreparedAnchor,
    local_boundary_stats,
    prepare_anchors,
    validate_inputs,
)
from src.pipelines.guidance.slicegan.condition import latent_preservation_weights
from src.pipelines.guidance.slicegan.generate import diffusion_references
from src.pipelines.guidance.slicegan.quality import (
    morphology_target,
    quality_passes,
    reference_stats,
)
from src.pipelines.guidance.slicegan.volume import calibrate, decode_frozen


class FakeSampler:
    def sample(self, shape):
        return torch.zeros(shape)


class CategoricalVAE(torch.nn.Module):
    def __init__(self, image_size: int = 64, latent_size: int = 16) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.latent_ch = 2
        self.num_phases = 2
        self.downsample_factor = image_size // latent_size

    def encode(self, image: torch.Tensor):
        latent = F.interpolate(
            image,
            size=(self.latent_size, self.latent_size),
            mode="nearest",
        ).repeat(1, self.latent_ch, 1, 1)
        return latent, torch.zeros_like(latent)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        score = F.interpolate(
            latent[:, :1],
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).sigmoid()
        return torch.cat([1.0 - score, score], dim=1)


class ParametricVAE(CategoricalVAE):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        score = F.interpolate(
            latent[:, :1] * self.scale,
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).sigmoid()
        return torch.cat([1.0 - score, score], dim=1)


class LatentSliceGANPipelineTest(unittest.TestCase):
    def test_validation_accepts_64_and_128_pixel_vaes(self):
        for image_size, latent_size in ((64, 16), (128, 32)):
            with self.subTest(image_size=image_size):
                vae = CategoricalVAE(image_size, latent_size)
                factor, output_latent = validate_inputs(
                    vae,
                    anchors=[
                        VolumeAnchor(
                            image=torch.zeros(image_size, image_size),
                            axis=0,
                            index=image_size // 2,
                        )
                    ],
                    target_fraction=None,
                    phase_fraction_tolerance=0.01,
                    volume_size=image_size,
                    num_phases=2,
                )

                self.assertEqual(factor, 4)
                self.assertEqual(output_latent, latent_size)

    def test_validation_accepts_larger_latent_output(self):
        vae = CategoricalVAE(64, 16)

        _, output_latent = validate_inputs(
            vae,
            anchors=[
                VolumeAnchor(
                    image=torch.zeros(64, 64),
                    axis=0,
                    index=64,
                    start=32,
                )
            ],
            target_fraction=torch.tensor([0.5, 0.5]),
            phase_fraction_tolerance=0.01,
            volume_size=128,
            num_phases=2,
        )

        self.assertEqual(output_latent, 32)

    def test_validation_rejects_latent_smaller_than_critic_input(self):
        vae = CategoricalVAE(64, 8)

        with self.assertRaisesRegex(ValueError, "at least 16"):
            validate_inputs(
                vae,
                anchors=[VolumeAnchor(torch.zeros(64, 64), 0, 32)],
                target_fraction=None,
                phase_fraction_tolerance=0.01,
                volume_size=64,
                num_phases=2,
            )

    def test_validation_rejects_fractional_anchor_labels(self):
        labels = torch.zeros(64, 64)
        labels[0, 0] = 0.5

        with self.assertRaisesRegex(ValueError, "integer phase values"):
            validate_inputs(
                CategoricalVAE(),
                anchors=[VolumeAnchor(labels, 0, 32)],
                target_fraction=None,
                phase_fraction_tolerance=0.01,
                volume_size=64,
                num_phases=2,
            )

    def test_anchor_preparation_encodes_base_size_patch(self):
        vae = CategoricalVAE()
        anchors = [VolumeAnchor(torch.ones(64, 64), 1, 32)]

        prepared = prepare_anchors(
            vae,
            anchors,
            factor=4,
            volume_size=64,
            num_phases=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(len(prepared[0].patches), 1)
        self.assertEqual(
            prepared[0].patches[0].latent.shape,
            torch.Size([2, 16, 16]),
        )
        self.assertEqual(prepared[0].patches[0].latent_index, 8)

    def test_diffusion_references_remain_in_latent_domain_for_critic(self):
        vae = CategoricalVAE()

        latents, images = diffusion_references(
            FakeSampler(),
            vae,
            count=3,
            num_phases=2,
        )

        self.assertEqual(latents.shape, torch.Size([3, 2, 16, 16]))
        self.assertEqual(images.shape, torch.Size([3, 2, 64, 64]))

    def test_anchor_decode_backpropagates_to_latent_without_vae_gradients(self):
        vae = ParametricVAE()
        latent = torch.randn(1, 2, 16, 16, requires_grad=True)

        probabilities = decode_frozen(vae, latent)
        probabilities[:, 1].mean().backward()

        self.assertIsNotNone(latent.grad)
        self.assertIsNone(vae.scale.grad)
        self.assertTrue(vae.scale.requires_grad)

    def test_morphology_target_uses_training_mixture_weight(self):
        anchor_labels = torch.zeros(1, 16, 16, dtype=torch.long)
        diffusion_labels = torch.arange(16).remainder(2).repeat(16, 1).unsqueeze(0)
        anchor = F.one_hot(anchor_labels, num_classes=2).movedim(-1, 1).float()
        diffusion = F.one_hot(diffusion_labels, num_classes=2).movedim(-1, 1).float()
        anchor_stats = reference_stats(anchor)
        diffusion_stats = reference_stats(diffusion)

        target = morphology_target(
            anchor,
            diffusion,
            mix_probability=0.1,
            target_fraction=torch.tensor([0.5, 0.5]),
        )

        expected_transition = (
            0.9 * anchor_stats["transition"]
            + 0.1 * diffusion_stats["transition"]
        )
        expected_run = (
            0.9 * anchor_stats["run_profile"]
            + 0.1 * diffusion_stats["run_profile"]
        )
        self.assertTrue(torch.allclose(target["transition"], expected_transition))
        self.assertTrue(torch.allclose(target["run_profile"], expected_run))

    def test_quality_gate_includes_transition_and_run_errors(self):
        stats = {
            "slicegan_quality_anchor_max_mismatch": torch.tensor(0.0),
            "slicegan_quality_phase_mae": torch.tensor(0.0),
            "slicegan_quality_transition_mae": torch.tensor(0.06),
            "slicegan_quality_run_mae": torch.tensor(0.0),
            "slicegan_quality_boundary_std": torch.tensor(0.0),
            "slicegan_quality_boundary_jump": torch.tensor(0.0),
        }

        self.assertFalse(
            quality_passes(
                stats,
                condition=SliceGANConditionConfig(),
                phase_fraction_tolerance=0.01,
            )
        )
        stats["slicegan_quality_transition_mae"] = torch.tensor(0.0)
        stats["slicegan_quality_run_mae"] = torch.tensor(0.06)
        self.assertFalse(
            quality_passes(
                stats,
                condition=SliceGANConditionConfig(),
                phase_fraction_tolerance=0.01,
            )
        )

    def test_generated_latent_slices_are_bounded_to_critic_size(self):
        volume = torch.randn(1, 4, 32, 32, 32)

        slices = sample_slices(volume, count=8, crop_size=16)

        self.assertEqual(slices.shape, torch.Size([8, 4, 16, 16]))

    def test_preservation_is_weak_near_anchor_and_strong_far_away(self):
        anchor = PreparedAnchor(
            labels=torch.zeros(64, 64),
            axis=0,
            index=32,
            start=0,
            patches=(),
        )

        weights = latent_preservation_weights(
            (16, 16, 16),
            [anchor],
            factor=4,
            sigma=8.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )[0, 0]

        self.assertLess(float(weights[8, 8, 8]), float(weights[0, 8, 8]))

    def test_boundary_stats_only_measure_anchor_footprint(self):
        volume = torch.zeros(8, 8, 8, dtype=torch.long)
        outside = torch.ones(8, 8, dtype=torch.bool)
        outside[3:5, 3:5] = False
        volume[4:, outside] = 1
        anchor = PreparedAnchor(
            labels=torch.zeros(2, 2, dtype=torch.long),
            axis=0,
            index=4,
            start=3,
            patches=(),
        )

        deviation, jump = local_boundary_stats(volume, anchor)

        self.assertEqual(float(deviation), 0.0)
        self.assertEqual(float(jump), 0.0)

    def test_calibration_protects_model_anchor_labels_not_target_copy(self):
        probabilities = torch.full((1, 2, 4, 4, 4), 0.5)
        probabilities[:, 0] = 0.9
        probabilities[:, 1] = 0.1
        anchor = PreparedAnchor(
            labels=torch.ones(2, 2, dtype=torch.long),
            axis=0,
            index=2,
            start=1,
            patches=(),
        )

        labels = calibrate(
            probabilities,
            [anchor],
            target_fraction=torch.tensor([0.5, 0.5]),
            num_phases=2,
        )

        self.assertTrue(torch.all(labels[2, 1:3, 1:3] == 0))

    def test_final_diagnostics_detect_repetition_and_global_cutoff(self):
        labels = torch.zeros(16, 16, 16, dtype=torch.long)
        labels[:, :, 1::2] = 1
        references = F.one_hot(
            labels[0],
            num_classes=2,
        ).movedim(-1, 0).unsqueeze(0).float()

        diagnostics = evaluate_phase_volume(
            labels,
            num_phases=2,
            references=references,
            target_fraction=torch.tensor([0.5, 0.5]),
            run_lengths=(2, 4, 8),
        )

        self.assertEqual(
            diagnostics["axis_run_profile"].shape,
            torch.Size([3, 2, 3]),
        )
        self.assertAlmostEqual(
            float(diagnostics["axis_exact_repeat_rate"][0]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["axis_exact_repeat_rate"][2]),
            0.0,
        )

        cutoff = torch.zeros(16, 16, 16, dtype=torch.long)
        cutoff[8:] = 1
        cutoff_diagnostics = evaluate_phase_volume(
            cutoff,
            num_phases=2,
            references=references,
            target_fraction=torch.tensor([0.5, 0.5]),
            run_lengths=(2, 4, 8),
        )
        self.assertAlmostEqual(
            float(cutoff_diagnostics["axis_global_boundary_jump"][0]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
