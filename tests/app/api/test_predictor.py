import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.app.api import (
    AnchorSlice,
    CriticConfig,
    JointConfig,
    PredictOptions,
    Predictor,
    PriorConfig,
    RefineConfig,
    ScaleConfig,
    TargetConfig,
)
from src.modeling.diffusion import DDPMProcess


class ZeroDenoiser(torch.nn.Module):
    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        phase_fractions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(latent)


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1
    num_phases = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        phase_one = torch.sigmoid(latent)
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class PredictOptionsTest(unittest.TestCase):
    def test_default_options_expose_one_joint_pipeline(self):
        options = PredictOptions(num_phases=2)

        self.assertIsInstance(options.joint, JointConfig)
        self.assertIsInstance(options.critic, CriticConfig)
        self.assertIsInstance(options.refine, RefineConfig)
        self.assertEqual(options.joint.steps, 0)
        self.assertEqual(options.critic.weight, 0.0)
        self.assertEqual(options.critic.mode, "score")
        self.assertTrue(options.refine.enabled)

    def test_loss_weights_accept_values_above_one(self):
        self.assertEqual(PriorConfig(weight=2.0).weight, 2.0)
        self.assertEqual(JointConfig(anchor_weight=2.0).anchor_weight, 2.0)
        self.assertEqual(JointConfig(axis_mass_weight=0.5).axis_mass_weight, 0.5)
        target = TargetConfig(
            slice_fraction_weight=2.0,
            global_fraction_weight=3.0,
        )
        self.assertEqual(target.slice_fraction_weight, 2.0)
        self.assertEqual(target.global_fraction_weight, 3.0)

    def test_joint_can_decode_all_planes_at_once(self):
        self.assertIsNone(JointConfig(decode_batch_size=None).decode_batch_size)

    def test_numeric_options_validate_types_and_ranges(self):
        invalid = (
            ("num_phases", lambda: PredictOptions(num_phases=2.5)),
            ("num_phases", lambda: PredictOptions(num_phases=257)),
            ("weight", lambda: PriorConfig(weight=-0.1)),
            ("anchor_strength", lambda: PriorConfig(anchor_strength=1.1)),
            ("learning_rate", lambda: JointConfig(learning_rate=0.0)),
            ("batch_size", lambda: JointConfig(batch_size=0)),
            ("decode_batch_size", lambda: JointConfig(decode_batch_size=0)),
            ("steps", lambda: JointConfig(steps=1.5)),
            ("axis_mass_weight", lambda: JointConfig(axis_mass_weight=-0.1)),
            ("progress", lambda: PredictOptions(num_phases=2, progress=1)),
            ("weight", lambda: CriticConfig(weight=-0.1)),
            ("mode", lambda: CriticConfig(mode="unknown")),
            ("enabled", lambda: RefineConfig(enabled=1)),
            ("decode_batch_size", lambda: ScaleConfig(decode_batch_size=True)),
            (
                "low_phase_conductivity",
                lambda: TargetConfig(
                    diffusivity_weight=1.0,
                    diffusivity_grid_size=2,
                    low_phase_conductivity=0.0,
                ),
            ),
        )
        for message, build in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()

    def test_scale_can_decode_all_planes_at_once(self):
        self.assertIsNone(ScaleConfig(decode_batch_size=None).decode_batch_size)

    def test_refine_can_be_disabled(self):
        config = RefineConfig(enabled=False)

        self.assertFalse(config.enabled)

    def test_phase_fractions_are_normalized_to_tuple(self):
        options = PredictOptions(
            num_phases=3,
            phase_fractions=[0.25, 0.15, 0.60],
        )

        self.assertEqual(options.phase_fractions, (0.25, 0.15, 0.60))

    def test_invalid_phase_fractions_are_rejected(self):
        cases = (
            ([0.5, 0.5], "one value per phase"),
            ([0.2, 0.2, 0.2], "sum to one"),
            ([0.5, float("nan"), 0.5], r"phase_fractions\[1\]"),
        )
        for fractions, message in cases:
            with self.subTest(fractions=fractions):
                with self.assertRaisesRegex(ValueError, message):
                    PredictOptions(num_phases=3, phase_fractions=fractions)


class PredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            DDPMProcess(timesteps=4),
            device="cpu",
        )

    def test_predict_returns_quantized_phase_volume(self):
        options = PredictOptions(
            num_phases=2,
            refine=RefineConfig(enabled=False),
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = self.predictor.predict(options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertFalse(bool(stats["refine_applied"]))
        self.assertIsInstance(stats["joint_history"], dict)

    def test_phase_fractions_require_joint_steps(self):
        with self.assertRaisesRegex(ValueError, "guidance steps"):
            self.predictor.predict(
                PredictOptions(num_phases=2, phase_fractions=(0.25, 0.75))
            )

    def test_joint_receives_fraction_and_prior_settings(self):
        options = PredictOptions(
            num_phases=2,
            phase_fractions=(0.25, 0.75),
            prior=PriorConfig(weight=0.6, t_min=0, t_max=3),
            joint=JointConfig(steps=1),
            targets=TargetConfig(global_fraction_weight=1.0),
            refine=RefineConfig(enabled=False),
        )
        latent = torch.zeros(1, 2, 2, 2)

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(
                latent,
                {"step": torch.tensor([1])},
            ),
        ) as optimize:
            self.predictor._run_joint(
                latent,
                options=options,
                anchors=None,
                target_labels=None,
                critic=None,
                t_max=3,
            )

        kwargs = optimize.call_args.kwargs
        self.assertTrue(
            torch.equal(
                kwargs["fraction_targets"],
                torch.tensor([0.25, 0.75]),
            )
        )
        self.assertEqual(kwargs["slice_fraction_weight"], 0.0)
        self.assertEqual(kwargs["global_fraction_weight"], 1.0)
        self.assertEqual(kwargs["sds_weight"], 0.6)
        self.assertEqual(kwargs["t_min"], 0)
        self.assertEqual(kwargs["t_max"], 3)
        self.assertTrue(kwargs["progress"])

    def test_reference_images_supply_global_fraction_when_not_explicit(self):
        options = PredictOptions(
            num_phases=2,
            targets=TargetConfig(global_fraction_weight=1.0),
        )
        labels = torch.tensor([[[0, 0], [0, 1]]])

        condition = self.predictor._resolve_fraction(options)
        diagnostic_target = self.predictor._resolve_target_fraction(options, labels)
        descriptor_targets = self.predictor._build_targets(options, labels)

        self.assertIsNone(condition)
        self.assertTrue(
            torch.equal(diagnostic_target, torch.tensor([0.75, 0.25]))
        )
        self.assertTrue(
            torch.allclose(
                descriptor_targets["fraction_targets"],
                torch.tensor([0.75, 0.25]),
                atol=1e-3,
            )
        )

    def test_anchor_supplies_global_fraction_without_duplicate_target_image(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            targets=TargetConfig(global_fraction_weight=1.0),
            refine=RefineConfig(enabled=False),
        )
        anchor = AnchorSlice(
            image=np.array([[0, 0], [0, 1]], dtype=np.uint8),
            axis=0,
            index=1,
        )
        latent = torch.zeros(1, 2, 2, 2)

        with (
            patch(
                "src.app.api.predictor.sample_latent",
                return_value=latent,
            ) as sample,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    latent,
                    {"step": torch.tensor([1])},
                ),
            ) as run_joint,
        ):
            self.predictor.predict(options, anchors=[anchor])

        labels = run_joint.call_args.kwargs["target_labels"]
        self.assertTrue(torch.equal(labels[0].cpu(), torch.from_numpy(anchor.image)))
        self.assertIsNone(sample.call_args.kwargs["phase_fractions"])
        self.assertIsNone(run_joint.call_args.kwargs["phase_fractions"])

    def test_tpc_reference_does_not_enable_fraction_conditioning(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            targets=TargetConfig(tpc_weight=1.0),
            refine=RefineConfig(enabled=False),
        )
        latent = torch.zeros(1, 2, 2, 2)

        with (
            patch(
                "src.app.api.predictor.sample_latent",
                return_value=latent,
            ) as sample,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(latent, {"step": torch.tensor([1])}),
            ) as run_joint,
        ):
            self.predictor.predict(
                options,
                target_images=[np.array([[0, 0], [0, 1]], dtype=np.uint8)],
            )

        self.assertIsNone(sample.call_args.kwargs["phase_fractions"])
        self.assertIsNone(run_joint.call_args.kwargs["phase_fractions"])

    def test_target_images_are_the_morphology_reference_when_anchor_is_present(self):
        options = PredictOptions(
            num_phases=2,
            phase_fractions=(0.5, 0.5),
            joint=JointConfig(steps=1),
            targets=TargetConfig(global_fraction_weight=1.0),
            refine=RefineConfig(enabled=False),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        target = np.ones((2, 2), dtype=np.uint8)
        latent = torch.zeros(1, 2, 2, 2)

        with (
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    latent,
                    {"step": torch.tensor([1])},
                ),
            ),
            patch(
                "src.app.api.predictor.evaluate_phase_volume",
                return_value={},
            ) as evaluate,
        ):
            self.predictor.predict(
                options,
                anchors=[anchor],
                target_images=[target],
            )

        references = evaluate.call_args.kwargs["references"]
        self.assertEqual(references.shape, (1, 2, 2, 2))
        self.assertTrue(torch.equal(references[:, 1], torch.ones(1, 2, 2)))

    def test_joint_receives_paper_descriptor_targets(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            targets=TargetConfig(
                tpc_weight=0.2,
                surface_area_weight=0.3,
                diffusivity_weight=0.4,
                diffusivity_grid_size=2,
                low_phase_conductivity=0.001,
            ),
        )
        latent = torch.zeros(1, 2, 2, 2)

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(
                latent,
                {"step": torch.tensor([1])},
            ),
        ) as optimize:
            self.predictor._run_joint(
                latent,
                options=options,
                anchors=None,
                target_labels=torch.zeros(1, 2, 2, dtype=torch.long),
                critic=None,
                t_max=3,
            )

        kwargs = optimize.call_args.kwargs
        self.assertEqual(kwargs["tpc_weight"], 0.2)
        self.assertEqual(kwargs["sa_weight"], 0.3)
        self.assertEqual(kwargs["diffusivity_weight"], 0.4)
        self.assertIsNotNone(kwargs["tpc_targets"])
        self.assertIsNotNone(kwargs["sa_targets"])
        self.assertIsNotNone(kwargs["diffusivity_targets"])
        self.assertIsNotNone(kwargs["diffusivity_solver"])

    def test_base_prediction_always_starts_from_lmpdd_latent(self):
        options = PredictOptions(
            num_phases=2,
            phase_fractions=(0.25, 0.75),
            joint=JointConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        latent = torch.zeros(1, 2, 2, 2)
        joint_history = {"step": torch.tensor([1])}

        with (
            patch(
                "src.app.api.predictor.sample_latent",
                return_value=latent,
            ) as sample,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(latent, joint_history),
            ) as run_joint,
        ):
            result, stats = self.predictor.predict(options)

        sample.assert_called_once()
        self.assertTrue(sample.call_args.kwargs["progress"])
        self.assertTrue(
            torch.equal(
                sample.call_args.kwargs["phase_fractions"],
                torch.tensor([0.25, 0.75]),
            )
        )
        run_joint.assert_called_once()
        self.assertTrue(
            torch.equal(
                run_joint.call_args.kwargs["phase_fractions"],
                torch.tensor([0.25, 0.75]),
            )
        )
        self.assertEqual(result.dtype, torch.uint8)
        self.assertIs(stats["joint_history"], joint_history)

    def test_base_lmpdd_receives_soft_anchor_condition(self):
        options = PredictOptions(
            num_phases=2,
            prior=PriorConfig(anchor_strength=0.25),
            joint=JointConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        latent = torch.zeros(1, 2, 2, 2)
        anchor_latent = torch.ones(2, 1, 2, 2)
        anchor_mask = torch.full_like(anchor_latent, 0.25)

        with (
            patch(
                "src.app.api.predictor.encode_anchors",
                return_value=(anchor_latent, anchor_mask),
            ) as encode,
            patch(
                "src.app.api.predictor.sample_latent",
                return_value=latent,
            ) as sample,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    latent,
                    {"step": torch.tensor([1])},
                ),
            ),
        ):
            self.predictor.predict(options, anchors=[anchor])

        self.assertEqual(encode.call_args.kwargs["peak_strength"], 0.25)
        self.assertIs(sample.call_args.kwargs["anchor_latent"], anchor_latent)
        self.assertIs(sample.call_args.kwargs["anchor_mask"], anchor_mask)

    def test_critic_requires_a_pretrained_gan_run(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            critic=CriticConfig(weight=0.1),
            refine=RefineConfig(enabled=False),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        with self.assertRaisesRegex(ValueError, "models.gan_run_dir"):
            self.predictor.predict(options, anchors=[anchor])

    def test_pretrained_critic_reaches_joint_without_fraction_condition(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            critic=CriticConfig(weight=0.1),
            refine=RefineConfig(enabled=False),
        )
        latent = torch.zeros(1, 2, 2, 2)
        volume = torch.zeros(2, 2, 2)
        critic = object()
        self.predictor.critic = critic

        with (
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    latent,
                    {"step": torch.tensor([1])},
                ),
            ) as joint,
        ):
            self.predictor.predict(options)

        self.assertIs(joint.call_args.kwargs["critic"], critic)
        self.assertNotIn("critic_fraction", joint.call_args.kwargs)

    def test_post_refine_is_skipped_after_base_critic_guidance(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            critic=CriticConfig(weight=0.1),
            refine=RefineConfig(enabled=True),
        )
        latent = torch.zeros(1, 2, 2, 2)
        self.predictor.critic = object()

        with (
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(latent, {"step": torch.tensor([1])}),
            ),
            patch("src.app.api.predictor.refine_probabilities") as refine,
            self.assertWarnsRegex(RuntimeWarning, "post-refine is skipped"),
        ):
            _, stats = self.predictor.predict(options)

        refine.assert_not_called()
        self.assertTrue(bool(stats["critic_enabled"]))
        self.assertFalse(bool(stats["refine_applied"]))

    def test_feature_critic_reaches_joint_with_categorical_references(self):
        options = PredictOptions(
            num_phases=2,
            critic=CriticConfig(weight=0.1, mode="feature"),
            joint=JointConfig(steps=1),
        )
        latent = torch.zeros(1, 2, 2, 2)
        labels = torch.tensor([[[0, 1], [1, 1]]])

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(latent, {"step": torch.tensor([1])}),
        ) as optimize:
            self.predictor._run_joint(
                latent,
                options=options,
                anchors=None,
                target_labels=labels,
                critic=object(),
                t_max=3,
            )

        kwargs = optimize.call_args.kwargs
        self.assertEqual(kwargs["critic_mode"], "feature")
        self.assertEqual(kwargs["critic_references"].shape, (1, 2, 2, 2))
        self.assertTrue(
            torch.equal(
                kwargs["critic_references"].argmax(dim=1),
                labels,
            )
        )

    def test_feature_critic_requires_target_or_anchor_reference(self):
        options = PredictOptions(
            num_phases=2,
            critic=CriticConfig(weight=0.1, mode="feature"),
            joint=JointConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        self.predictor.critic = object()

        with self.assertRaisesRegex(ValueError, "target_images or anchors"):
            self.predictor.predict(options)

    def test_large_prediction_uses_scale_refinement_not_base_joint(self):
        options = PredictOptions(
            num_phases=2,
            critic=CriticConfig(weight=0.1),
            scale=ScaleConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        generated = torch.zeros(1, 4, 4, 4)
        self.predictor.critic = object()

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(generated, {"volume_size": 4}),
            ) as generate,
            patch.object(
                self.predictor,
                "_refine_large",
                return_value=(
                    generated,
                    {"step": torch.tensor([1])},
                ),
            ) as refine,
        ):
            volume, stats = self.predictor.predict(options, volume_size=4)

        generate.assert_called_once()
        refine.assert_called_once()
        self.assertNotIn("critic_fraction", refine.call_args.kwargs)
        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("final_phase_fraction", stats)

    def test_post_refine_is_skipped_after_scale_critic_guidance(self):
        options = PredictOptions(
            num_phases=2,
            critic=CriticConfig(weight=0.1),
            scale=ScaleConfig(steps=1),
            refine=RefineConfig(enabled=True),
        )
        latent = torch.zeros(1, 4, 4, 4)
        self.predictor.critic = object()

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(latent, {"volume_size": 4}),
            ),
            patch.object(
                self.predictor,
                "_refine_large",
                return_value=(latent, {"step": torch.tensor([1])}),
            ),
            patch(
                "src.app.api.predictor.refine_large_probabilities"
            ) as refine,
            self.assertWarnsRegex(RuntimeWarning, "post-refine is skipped"),
        ):
            _, stats = self.predictor.predict(options, volume_size=4)

        refine.assert_not_called()
        self.assertTrue(bool(stats["critic_enabled"]))
        self.assertFalse(bool(stats["refine_applied"]))

    def test_feature_critic_references_are_tiled_for_scale_guidance(self):
        options = PredictOptions(
            num_phases=2,
            critic=CriticConfig(weight=0.1, mode="feature"),
            scale=ScaleConfig(steps=1),
        )
        latent = torch.zeros(1, 4, 4, 4)
        labels = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]]
        )
        self.predictor.critic = object()

        with patch(
            "src.app.api.predictor.optimize_large_latent",
            return_value=(latent, {"step": torch.tensor([1])}),
        ) as optimize:
            self.predictor._refine_large(
                latent,
                options=options,
                anchors=None,
                target_labels=labels,
                descriptor_tile_size=2,
                t_max=3,
            )

        kwargs = optimize.call_args.kwargs
        self.assertEqual(kwargs["critic_mode"], "feature")
        self.assertEqual(kwargs["critic_references"].shape, (4, 2, 2, 2))
        self.assertTrue(
            torch.equal(
                kwargs["critic_references"].sum(dim=1),
                torch.ones(4, 2, 2),
            )
        )

    def test_large_full_size_tpc_target_is_tiled_to_vae_size(self):
        options = PredictOptions(
            num_phases=2,
            progress=False,
            scale=ScaleConfig(steps=1),
            targets=TargetConfig(tpc_weight=1.0),
            refine=RefineConfig(enabled=False),
        )
        generated = torch.zeros(1, 4, 4, 4)

        with patch.object(
            self.predictor,
            "_generate_large",
            return_value=(generated, {"volume_size": 4}),
        ):
            volume, stats = self.predictor.predict(
                options,
                target_images=[np.zeros((4, 4), dtype=np.uint8)],
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(stats["scale_history"]["tpc"].shape, torch.Size([1]))

    def test_large_anchor_and_target_keep_reference_roles_separate(self):
        options = PredictOptions(
            num_phases=2,
            targets=TargetConfig(global_fraction_weight=1.0),
            scale=ScaleConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        target = np.ones((4, 4), dtype=np.uint8)
        generated = torch.zeros(1, 4, 4, 4)
        probabilities = torch.full((1, 2, 4, 4, 4), 0.5)
        selected = torch.zeros(4, 4, 4)

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(generated, {"volume_size": 4}),
            ),
            patch.object(
                self.predictor,
                "_refine_large",
                return_value=(
                    generated,
                    {"step": torch.tensor([1])},
                ),
            ) as refine,
            patch(
                "src.app.api.predictor.decode_large_volume_probabilities",
                return_value=probabilities,
            ),
            patch(
                "src.app.api.predictor.refine_large_probabilities",
                return_value=probabilities,
            ),
            patch(
                "src.app.api.predictor.evaluate_phase_volume",
                return_value={},
            ) as evaluate,
        ):
            self.predictor.predict(
                options,
                anchors=[anchor],
                target_images=[target],
                volume_size=4,
            )

        self.assertEqual(refine.call_args.kwargs["target_labels"].shape, (1, 4, 4))
        self.assertEqual(evaluate.call_args.kwargs["references"].shape, (1, 2, 4, 4))

    def test_large_prediction_warns_when_base_guidance_is_configured(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            refine=RefineConfig(enabled=False),
        )
        generated = torch.zeros(1, 4, 4, 4)

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(generated, {"volume_size": 4}),
            ),
            self.assertWarnsRegex(RuntimeWarning, "base-size only"),
        ):
            self.predictor.predict(options, volume_size=4)


if __name__ == "__main__":
    unittest.main()
