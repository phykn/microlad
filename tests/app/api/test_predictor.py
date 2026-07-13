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
    QualityConfig,
    RefineConfig,
    ScaleConfig,
    TargetConfig,
)
from src.modeling.diffusion import DDPMProcess


class ZeroDenoiser(torch.nn.Module):
    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
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
        self.assertIsInstance(options.quality, QualityConfig)
        self.assertEqual(options.joint.steps, 0)
        self.assertEqual(options.critic.steps, 0)

    def test_loss_weights_accept_values_above_one(self):
        self.assertEqual(PriorConfig(weight=2.0).weight, 2.0)
        self.assertEqual(JointConfig(anchor_weight=2.0).anchor_weight, 2.0)
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
            ("progress", lambda: PredictOptions(num_phases=2, progress=1)),
            ("candidate_count", lambda: CriticConfig(candidate_count=0)),
            ("slice_steps", lambda: ScaleConfig(slice_steps=True)),
            ("calibration_budget", lambda: QualityConfig(calibration_budget=1.1)),
        )
        for message, build in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()

    def test_critic_requires_joint_optimization(self):
        with self.assertRaisesRegex(ValueError, "critic.steps"):
            PredictOptions(num_phases=2, critic=CriticConfig(steps=1))
        with self.assertRaisesRegex(ValueError, "critic.weight"):
            PredictOptions(num_phases=2, critic=CriticConfig(weight=0.1))

    def test_scale_flags_require_booleans(self):
        with self.assertRaisesRegex(ValueError, "balanced_slices"):
            ScaleConfig(balanced_slices=1)

    def test_refine_candidates_are_unique_and_ordered(self):
        config = RefineConfig(candidates=(0, 2, 0, 1))

        self.assertEqual(config.candidates, (0, 2, 1))

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
            refine=RefineConfig(candidates=(0,)),
            quality=QualityConfig(
                anchor_tolerance=1.0,
                morphology_tolerance=1.0,
                continuity_tolerance=1.0,
                repeat_tolerance=1.0,
                calibration_budget=1.0,
            ),
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = self.predictor.predict(options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertEqual(int(stats["candidate_count"]), 1)

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
            refine=RefineConfig(candidates=(0,)),
        )
        latent = torch.zeros(1, 2, 2, 2)

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(
                (latent,),
                {
                    "joint_steps": torch.tensor(1),
                    "joint_candidate_steps": torch.tensor([1]),
                },
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

        target = self.predictor._resolve_fraction(options, labels)

        self.assertTrue(torch.equal(target, torch.tensor([0.75, 0.25])))

    def test_anchor_supplies_global_fraction_without_duplicate_target_image(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            targets=TargetConfig(global_fraction_weight=1.0),
            refine=RefineConfig(candidates=(0,)),
        )
        anchor = AnchorSlice(
            image=np.array([[0, 0], [0, 1]], dtype=np.uint8),
            axis=0,
            index=1,
        )
        latent = torch.zeros(1, 2, 2, 2)

        with (
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    (latent,),
                    {
                        "joint_steps": torch.tensor(1),
                        "joint_candidate_steps": torch.tensor([0]),
                    },
                ),
            ) as run_joint,
            patch(
                "src.app.api.predictor.select_latent_volume",
                return_value=(
                    torch.zeros(2, 2, 2),
                    {"quality_passed": torch.tensor(True)},
                ),
            ),
        ):
            self.predictor.predict(options, anchors=[anchor])

        labels = run_joint.call_args.kwargs["target_labels"]
        self.assertTrue(torch.equal(labels[0].cpu(), torch.from_numpy(anchor.image)))

    def test_target_images_are_the_morphology_reference_when_anchor_is_present(self):
        options = PredictOptions(
            num_phases=2,
            phase_fractions=(0.5, 0.5),
            joint=JointConfig(steps=1),
            targets=TargetConfig(global_fraction_weight=1.0),
            refine=RefineConfig(candidates=(0,)),
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
                    (latent,),
                    {
                        "joint_steps": torch.tensor(1),
                        "joint_candidate_steps": torch.tensor([0]),
                    },
                ),
            ),
            patch(
                "src.app.api.predictor.select_latent_volume",
                return_value=(
                    torch.zeros(2, 2, 2),
                    {"quality_passed": torch.tensor(True)},
                ),
            ) as select,
        ):
            self.predictor.predict(
                options,
                anchors=[anchor],
                target_images=[target],
            )

        references = select.call_args.kwargs["references"]
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
            ),
        )
        latent = torch.zeros(1, 2, 2, 2)

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(
                (latent,),
                {
                    "joint_steps": torch.tensor(1),
                    "joint_candidate_steps": torch.tensor([1]),
                },
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
            joint=JointConfig(steps=1),
            refine=RefineConfig(candidates=(0,)),
        )
        latent = torch.zeros(1, 2, 2, 2)
        volume = torch.zeros(2, 2, 2)
        joint_stats = {
            "joint_steps": torch.tensor(1),
            "joint_candidate_steps": torch.tensor([0]),
        }

        with (
            patch(
                "src.app.api.predictor.sample_latent",
                return_value=latent,
            ) as sample,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=((latent,), joint_stats),
            ) as run_joint,
            patch(
                "src.app.api.predictor.select_latent_volume",
                return_value=(
                    volume,
                    {
                        "candidate_count": torch.tensor(1),
                        "quality_passed": torch.tensor(False),
                    },
                ),
            ),
        ):
            with self.assertWarnsRegex(RuntimeWarning, "least-violation"):
                result, stats = self.predictor.predict(options)

        sample.assert_called_once()
        self.assertTrue(sample.call_args.kwargs["progress"])
        run_joint.assert_called_once()
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(int(stats["joint_steps"]), 1)

    def test_base_lmpdd_receives_soft_anchor_condition(self):
        options = PredictOptions(
            num_phases=2,
            prior=PriorConfig(anchor_strength=0.25),
            joint=JointConfig(steps=1),
            refine=RefineConfig(candidates=(0,)),
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
                    (latent,),
                    {
                        "joint_steps": torch.tensor(1),
                        "joint_candidate_steps": torch.tensor([0]),
                    },
                ),
            ),
            patch(
                "src.app.api.predictor.select_latent_volume",
                return_value=(torch.zeros(2, 2, 2), {}),
            ),
        ):
            self.predictor.predict(options, anchors=[anchor])

        self.assertEqual(encode.call_args.kwargs["peak_strength"], 0.25)
        self.assertIs(sample.call_args.kwargs["anchor_latent"], anchor_latent)
        self.assertIs(sample.call_args.kwargs["anchor_mask"], anchor_mask)

    def test_critic_uses_anchor_or_target_references(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            critic=CriticConfig(steps=1, weight=0.1),
            refine=RefineConfig(candidates=(0,)),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        latent = torch.zeros(1, 2, 2, 2)
        volume = torch.zeros(2, 2, 2)

        with (
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch.object(
                self.predictor,
                "_fit_critic",
                return_value=(None, {"critic_steps": torch.tensor(1)}),
            ) as train,
            patch.object(
                self.predictor,
                "_run_joint",
                return_value=(
                    (latent,),
                    {
                        "joint_steps": torch.tensor(1),
                        "joint_candidate_steps": torch.tensor([0]),
                    },
                ),
            ),
            patch(
                "src.app.api.predictor.select_latent_volume",
                return_value=(volume, {}),
            ),
        ):
            self.predictor.predict(options, anchors=[anchor])

        train.assert_called_once()
        references = train.call_args.args[1]
        self.assertEqual(references.shape, torch.Size([1, 2, 2]))

    def test_failed_critic_validation_disables_only_critic_guidance(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            critic=CriticConfig(steps=1, weight=0.1),
            refine=RefineConfig(candidates=(0,)),
        )
        latent = torch.zeros(1, 2, 2, 2)
        failed = {
            "critic_validation_accuracy": torch.tensor(0.8),
            "critic_damage_accuracy": torch.tensor(0.8),
            "critic_shuffle_accuracy": torch.tensor(0.8),
            "critic_validation_margin": torch.tensor(0.0),
            "critic_input_gradient_finite": torch.tensor(True),
        }

        with (
            patch(
                "src.app.api.predictor.encode_refs",
                return_value=torch.zeros(1, 8, 1, 2, 2),
            ),
            patch("src.app.api.predictor.sample_latent", return_value=latent),
            patch("src.app.api.predictor.train_critic", return_value=failed),
        ):
            critic, stats = self.predictor._fit_critic(
                latent,
                torch.zeros(1, 2, 2),
                options=options,
            )

        self.assertIsNone(critic)
        self.assertFalse(bool(stats["critic_enabled"]))

        with patch(
            "src.app.api.predictor.optimize_latent",
            return_value=(
                (latent,),
                {
                    "joint_steps": torch.tensor(1),
                    "joint_candidate_steps": torch.tensor([0]),
                },
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

        self.assertEqual(optimize.call_args.kwargs["critic_weight"], 0.0)

    def test_large_prediction_uses_scale_refinement_not_base_joint(self):
        options = PredictOptions(
            num_phases=2,
            scale=ScaleConfig(steps=1),
            refine=RefineConfig(candidates=(0,)),
        )
        generated = torch.zeros(4, 4, 4)

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(generated, {"volume_size": 4}),
            ) as generate,
            patch.object(
                self.predictor,
                "_refine_large",
                return_value=(generated, {"history_loss": torch.tensor(0.0)}),
            ) as refine,
            self.assertWarnsRegex(RuntimeWarning, "least-violation"),
        ):
            volume, stats = self.predictor.predict(options, volume_size=4)

        generate.assert_called_once()
        refine.assert_called_once()
        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("final_phase_fraction", stats)

    def test_large_anchor_and_target_keep_reference_roles_separate(self):
        options = PredictOptions(
            num_phases=2,
            targets=TargetConfig(global_fraction_weight=1.0),
            scale=ScaleConfig(steps=1),
            refine=RefineConfig(candidates=(0,)),
        )
        anchor = AnchorSlice(
            image=np.zeros((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        target = np.ones((4, 4), dtype=np.uint8)
        generated = torch.zeros(4, 4, 4)

        with (
            patch.object(
                self.predictor,
                "_generate_large",
                return_value=(generated, {"volume_size": 4}),
            ),
            patch.object(
                self.predictor,
                "_refine_large",
                return_value=(generated, {"history_loss": torch.tensor(0.0)}),
            ) as refine,
            patch(
                "src.app.api.predictor.refine_large_candidates",
                return_value=(generated,),
            ),
            patch(
                "src.app.api.predictor.select_label_volume",
                return_value=(
                    generated,
                    {"quality_passed": torch.tensor(True)},
                ),
            ) as select,
        ):
            self.predictor.predict(
                options,
                anchors=[anchor],
                target_images=[target],
                volume_size=4,
            )

        self.assertEqual(refine.call_args.kwargs["target_labels"].shape, (1, 4, 4))
        self.assertEqual(select.call_args.kwargs["references"].shape, (1, 2, 4, 4))

    def test_large_prediction_warns_when_base_guidance_is_configured(self):
        options = PredictOptions(
            num_phases=2,
            joint=JointConfig(steps=1),
            refine=RefineConfig(candidates=(0,)),
        )
        generated = torch.zeros(4, 4, 4)

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
