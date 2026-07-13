import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.app.api import (
    AnchorConfig,
    AnchorSlice,
    JointConfig,
    PredictOptions,
    Predictor,
    RefineConfig,
    SDSConfig,
    SliceGANConditionConfig,
    SliceGANConfig,
    SliceGANTrainConfig,
    TargetConfig,
)


class IdentityDDPM:
    def __init__(self, timesteps: int = 4) -> None:
        self.num_timesteps = timesteps
        self.posterior_variance = torch.zeros(timesteps)
        self.steps: list[int] = []

    def p_mean(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.steps.append(int(t[0].item()))
        return x

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.p_mean(model, x, t)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values.to(device=t.device)[t].view(shape)


class ZeroDenoiser(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class IdentityVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 2
        self.latent_size = 2
        self.latent_ch = 1
        self.downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clone()


class ZeroDownsampleVAE(IdentityVAE):
    def __init__(self) -> None:
        super().__init__()
        self.downsample_factor = 0


def fast_slicegan_config(*, intersection_tolerance: float = 0.1) -> SliceGANConfig:
    return SliceGANConfig(
        training=SliceGANTrainConfig(steps=1),
        intersection_tolerance=intersection_tolerance,
    )


class PredictOptionsTest(unittest.TestCase):
    def test_predict_options_accepts_grouped_slicegan_config(self):
        config = SliceGANConfig(
            training=SliceGANTrainConfig(steps=12, hybrid_steps=3),
            conditioning=SliceGANConditionConfig(steps=7, finetune_steps=2),
            intersection_tolerance=0.05,
        )

        options = PredictOptions(num_phases=2, slicegan=config)

        self.assertIs(options.slicegan, config)
        self.assertEqual(options.slicegan.training.steps, 12)
        self.assertEqual(options.slicegan.conditioning.steps, 7)

    def test_predict_options_rejects_non_integer_num_phases(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            PredictOptions(num_phases=2.5)

    def test_predict_options_rejects_num_phases_that_exceed_uint8_range(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            PredictOptions(num_phases=257)

    def test_predict_options_rejects_weights_outside_zero_to_one(self):
        with self.assertRaisesRegex(ValueError, "weight"):
            SDSConfig(weight=1.1)
        with self.assertRaisesRegex(ValueError, "weight"):
            AnchorConfig(weight=-0.1)
        with self.assertRaisesRegex(ValueError, "vf_weight"):
            TargetConfig(vf_weight=2.0)

    def test_predict_options_rejects_non_finite_numeric_values(self):
        cases = (
            ("weight", lambda: SDSConfig(weight=float("nan"))),
            ("weight", lambda: AnchorConfig(weight=float("nan"))),
            (
                "low_conductivity",
                lambda: TargetConfig(low_conductivity=float("nan")),
            ),
            ("learning_rate", lambda: SDSConfig(learning_rate=float("nan"))),
        )

        for message, build in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()

    def test_predict_options_rejects_invalid_sds_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            SDSConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            SDSConfig(batch_size=1.5)

    def test_predict_options_rejects_non_boolean_balanced_slices(self):
        with self.assertRaisesRegex(ValueError, "balanced_slices"):
            SDSConfig(balanced_slices=1)
        with self.assertRaisesRegex(ValueError, "consensus"):
            SDSConfig(consensus=1)

    def test_predict_options_rejects_non_integer_step_counts(self):
        cases = (
            ("steps", lambda: SDSConfig(steps=1.5)),
            ("slice_steps", lambda: SDSConfig(slice_steps=True)),
            ("t_min", lambda: SDSConfig(t_min=1.5)),
            ("t_max", lambda: SDSConfig(t_max=2.5)),
            ("steps", lambda: RefineConfig(steps=1.5)),
            ("fit_steps", lambda: AnchorConfig(fit_steps=1.5)),
            ("slab_radius", lambda: AnchorConfig(slab_radius=1.5)),
            ("steps", lambda: JointConfig(steps=1.5)),
            ("batch_size", lambda: JointConfig(batch_size=1.5)),
            ("steps", lambda: SliceGANTrainConfig(steps=1.5)),
            ("steps", lambda: SliceGANConditionConfig(steps=1.5)),
            ("seed", lambda: SliceGANConfig(seed=1.5)),
        )

        for message, build in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, f"{message}.*integer"):
                    build()

    def test_predict_options_rejects_invalid_anchor_fit_lr(self):
        with self.assertRaisesRegex(ValueError, "fit_lr"):
            AnchorConfig(fit_lr=0.0)

    def test_predict_options_rejects_invalid_anchor_slab_values(self):
        with self.assertRaisesRegex(ValueError, "slab_radius"):
            AnchorConfig(slab_radius=-1)
        with self.assertRaisesRegex(ValueError, "slab_weight"):
            AnchorConfig(slab_weight=1.1)

    def test_predict_options_rejects_invalid_joint_values(self):
        with self.assertRaisesRegex(ValueError, "axis_consensus"):
            AnchorConfig(axis_consensus=1)
        with self.assertRaisesRegex(ValueError, "latent_sigma"):
            AnchorConfig(latent_sigma=-1.0)
        with self.assertRaisesRegex(ValueError, "latent_strength"):
            AnchorConfig(latent_strength=0.0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            JointConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            JointConfig(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "entropy_weight"):
            JointConfig(entropy_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "transition_weight"):
            JointConfig(transition_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "discriminator_lr"):
            JointConfig(discriminator_lr=0.0)
        with self.assertRaisesRegex(ValueError, "cannot both"):
            PredictOptions(
                num_phases=2,
                joint=JointConfig(steps=1),
                sds=SDSConfig(steps=1),
            )
        with self.assertRaisesRegex(ValueError, "replaces refine"):
            PredictOptions(
                num_phases=2,
                joint=JointConfig(steps=1),
                refine=RefineConfig(steps=1),
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            PredictOptions(
                num_phases=2,
                slicegan=SliceGANConfig(),
                joint=JointConfig(steps=1),
            )

    def test_slicegan_config_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            SliceGANTrainConfig(steps=-1)
        with self.assertRaisesRegex(ValueError, "hybrid_steps"):
            SliceGANTrainConfig(hybrid_steps=-1)
        with self.assertRaisesRegex(ValueError, "steps"):
            SliceGANConditionConfig(steps=-1)
        with self.assertRaisesRegex(ValueError, "seed"):
            SliceGANConfig(seed=-1)

    def test_predict_options_accepts_and_normalizes_phase_fractions(self):
        options = PredictOptions(
            num_phases=3,
            phase_fractions=[0.25, 0.15, 0.60],
        )

        self.assertEqual(options.phase_fractions, (0.25, 0.15, 0.60))

    def test_predict_options_rejects_invalid_phase_fractions(self):
        cases = [
            ([0.5, 0.5], "one value per phase"),
            ([0.2, 0.2, 0.2], "sum to one"),
            ([0.5, float("nan"), 0.5], r"phase_fractions\[1\]"),
            ([1.1, 0.0, -0.1], "between 0 and 1"),
        ]
        for fractions, message in cases:
            with self.subTest(fractions=fractions):
                with self.assertRaisesRegex(ValueError, message):
                    PredictOptions(num_phases=3, phase_fractions=fractions)

    def test_predict_options_accepts_default_phase_fraction_tolerance(self):
        options = PredictOptions(num_phases=3)

        self.assertEqual(options.phase_fraction_tolerance, 0.01)

    def test_predict_options_rejects_invalid_phase_fraction_tolerance(self):
        for tolerance in (-0.01, 1.01, float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "phase_fraction_tolerance"):
                    PredictOptions(
                        num_phases=3,
                        phase_fraction_tolerance=tolerance,
                    )

    def test_predict_options_rejects_invalid_intersection_tolerance(self):
        for tolerance in (-0.1, 1.1, float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "intersection_tolerance"):
                    PredictOptions(
                        num_phases=2,
                        slicegan=SliceGANConfig(intersection_tolerance=tolerance),
                    )


class PredictorTest(unittest.TestCase):
    def test_balanced_schedule_visits_every_slice_once_per_sweep(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )

        schedule = predictor._build_balanced_schedule(
            steps=6,
            batch_size=1,
            volume_size=2,
        )

        self.assertEqual(len(schedule), 6)
        self.assertEqual(
            set(schedule), {(axis, index) for axis in range(3) for index in range(2)}
        )

    def test_predict_returns_quantized_phase_volume(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options=options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_predict_accepts_options_as_first_argument(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIsInstance(stats, dict)

    def test_predict_builds_targets_and_runs_sds_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(
            num_phases=2,
            sds=SDSConfig(steps=1, slice_steps=1, t_min=1, t_max=3, weight=0.0),
            targets=TargetConfig(vf_weight=1.0),
        )
        target_images = [np.zeros((2, 2), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                target_images=target_images,
                options=options,
            )

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIn("vf", stats)
        self.assertIn("loss", stats)

    def test_predict_routes_joint_3d_optimization_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(num_phases=2, joint=JointConfig(steps=1))
        joint_volume = torch.ones(2, 2, 2)

        with (
            patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)),
            patch.object(
                predictor,
                "_run_joint_3d",
                return_value=(joint_volume, {"joint_steps": torch.tensor(1)}),
            ) as run_joint,
        ):
            volume, stats = predictor.predict(options)

        run_joint.assert_called_once()
        self.assertTrue(torch.equal(volume, torch.ones(2, 2, 2, dtype=torch.uint8)))
        self.assertEqual(int(stats["joint_steps"]), 1)

    def test_predict_routes_conditional_slicegan_without_generating_base(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(num_phases=2, slicegan=fast_slicegan_config())
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )
        generated = torch.ones(2, 2, 2)

        with (
            patch.object(
                predictor,
                "_run_slicegan",
                return_value=(generated, {"slicegan_steps": torch.tensor(1)}),
            ) as run_slicegan,
            patch.object(predictor, "_generate_base") as generate_base,
        ):
            volume, stats = predictor.predict(options, anchors=[anchor])

        run_slicegan.assert_called_once()
        generate_base.assert_not_called()
        self.assertTrue(torch.equal(volume, torch.ones(2, 2, 2, dtype=torch.uint8)))
        self.assertEqual(int(stats["slicegan_steps"]), 1)

    def test_run_slicegan_passes_multiple_axes_and_phase_fraction(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(),
            device="cpu",
        )
        predictor.vae.image_size = 64
        image = np.zeros((64, 64), dtype=np.uint8)
        anchors = [
            AnchorSlice(image=image, axis=0, index=32),
            AnchorSlice(image=image.copy(), axis=1, index=32),
        ]
        options = PredictOptions(
            num_phases=2,
            slicegan=fast_slicegan_config(intersection_tolerance=0.05),
            phase_fractions=(0.4, 0.6),
        )

        with patch(
            "src.app.api.predictor.generate_conditional_slicegan",
            return_value=(torch.zeros(64, 64, 64), {}),
        ) as generate:
            predictor._run_slicegan(64, options=options, anchors=anchors)

        kwargs = generate.call_args.kwargs
        self.assertEqual(
            [(target.axis, target.index) for target in kwargs["anchors"]],
            [(0, 32), (1, 32)],
        )
        self.assertTrue(
            torch.allclose(kwargs["target_fraction"], torch.tensor([0.4, 0.6]))
        )
        self.assertEqual(kwargs["config"].intersection_tolerance, 0.05)
        self.assertEqual(kwargs["phase_fraction_tolerance"], 0.01)
        self.assertEqual(kwargs["volume_size"], 64)

    def test_run_slicegan_preserves_absolute_index_in_scaled_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(),
            device="cpu",
        )
        predictor.vae.image_size = 64
        image = np.zeros((64, 64), dtype=np.uint8)
        anchors = [AnchorSlice(image=image, axis=2, index=84)]
        options = PredictOptions(num_phases=2, slicegan=fast_slicegan_config())

        with patch(
            "src.app.api.predictor.generate_conditional_slicegan",
            return_value=(torch.zeros(128, 128, 128), {}),
        ) as generate:
            predictor._run_slicegan(128, options=options, anchors=anchors)

        kwargs = generate.call_args.kwargs
        target = kwargs["anchors"][0]
        self.assertEqual((target.axis, target.index, target.start), (2, 84, 32))
        self.assertEqual(kwargs["volume_size"], 128)

    def test_predict_accepts_multiple_same_axis_absolute_scale_anchors(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(),
            device="cpu",
        )
        predictor.vae.image_size = 64
        image = np.zeros((64, 64), dtype=np.uint8)
        anchors = [
            AnchorSlice(image=image, axis=0, index=20),
            AnchorSlice(image=image.copy(), axis=0, index=100),
        ]
        options = PredictOptions(num_phases=2, slicegan=fast_slicegan_config())

        with patch(
            "src.app.api.predictor.generate_conditional_slicegan",
            return_value=(torch.zeros(128, 128, 128), {}),
        ) as generate:
            volume, _ = predictor.predict(
                options,
                anchors=anchors,
                volume_size=128,
            )

        targets = generate.call_args.kwargs["anchors"]
        self.assertEqual(
            [(target.axis, target.index, target.start) for target in targets],
            [(0, 20, 32), (0, 100, 32)],
        )
        self.assertEqual(volume.shape, torch.Size([128, 128, 128]))

    def test_predict_rejects_scale_anchor_index_outside_output_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(),
            device="cpu",
        )
        predictor.vae.image_size = 64
        anchor = AnchorSlice(
            image=np.zeros((64, 64), dtype=np.uint8),
            axis=1,
            index=128,
        )

        with self.assertRaisesRegex(ValueError, "output axis"):
            predictor.predict(
                PredictOptions(num_phases=2, slicegan=fast_slicegan_config()),
                anchors=[anchor],
                volume_size=128,
            )

    def test_joint_prediction_applies_anchor_only_in_image_space(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with (
            patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)),
            patch(
                "src.app.api.predictor.encode_anchors",
                return_value=(None, None),
            ) as encode,
            patch.object(
                predictor,
                "_run_joint_3d",
                return_value=(torch.zeros(2, 2, 2), {}),
            ),
        ):
            predictor.predict(
                anchors=[anchor],
                options=PredictOptions(num_phases=2, joint=JointConfig(steps=1)),
            )

        self.assertIsNone(encode.call_args.args[1])

    def test_predict_blends_anchor_latent_without_forced_overwrite(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                anchors=[anchor],
                options=PredictOptions(num_phases=2),
            )

        self.assertTrue(torch.equal(volume[1], torch.ones(2, 2, dtype=torch.uint8)))
        self.assertTrue(torch.equal(volume[0], torch.zeros(2, 2, dtype=torch.uint8)))
        self.assertIsInstance(stats, dict)

    def test_predict_fits_anchor_after_refinement_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                anchors=[anchor],
                options=PredictOptions(
                    num_phases=2,
                    refine=RefineConfig(steps=1),
                    anchor=AnchorConfig(fit_steps=2, fit_lr=0.1),
                ),
            )

        self.assertTrue(torch.equal(volume[1], torch.ones(2, 2, dtype=torch.uint8)))
        self.assertIn("anchor_fit", stats)

    def test_predict_requires_anchor_when_anchor_fit_is_enabled(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            with self.assertRaisesRegex(ValueError, "anchors"):
                predictor.predict(
                    PredictOptions(
                        num_phases=2,
                        anchor=AnchorConfig(fit_steps=1),
                    ),
                )

    def test_predict_lmpdd_anchor_axis_is_stable_for_short_schedules(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=2),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, _ = predictor.predict(
                anchors=[anchor],
                options=PredictOptions(num_phases=2),
            )

        self.assertTrue(torch.equal(volume[1], torch.ones(2, 2, dtype=torch.uint8)))
        self.assertTrue(torch.equal(volume[0], torch.zeros(2, 2, dtype=torch.uint8)))

    def test_predict_rejects_target_loss_without_target_images(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(
            num_phases=2,
            sds=SDSConfig(steps=1),
            targets=TargetConfig(vf_weight=1.0),
        )

        with self.assertRaisesRegex(ValueError, "target_images"):
            predictor.predict(options)

    def test_predict_rejects_small_volume_target_images_with_wrong_size(self):
        predictor = Predictor(
            IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu"
        )
        options = PredictOptions(
            num_phases=2,
            sds=SDSConfig(steps=1, weight=0.0),
            targets=TargetConfig(surface_weight=1.0),
        )
        target_images = [np.zeros((4, 4), dtype=np.uint8)]

        with self.assertRaisesRegex(ValueError, "target images"):
            predictor.predict(options, target_images=target_images)

    def test_predict_accepts_exclusive_sds_t_max_equal_to_num_timesteps(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                PredictOptions(
                    num_phases=2,
                    sds=SDSConfig(
                        steps=1,
                        slice_steps=1,
                        t_min=1,
                        t_max=4,
                        weight=0.0,
                    ),
                ),
            )

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIn("steps", stats)

    def test_predict_uses_volume_size_for_large_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2),
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_predict_rejects_non_integer_volume_size(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "volume_size.*integer"):
            predictor.predict(PredictOptions(num_phases=2), volume_size=4.5)

    def test_predict_rejects_invalid_vae_downsample_factor_before_sampling(self):
        predictor = Predictor(
            ZeroDownsampleVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", side_effect=AssertionError("sampling started")):
            with self.assertRaisesRegex(ValueError, "downsample"):
                predictor.predict(PredictOptions(num_phases=2), volume_size=4)

    def test_predict_uses_full_size_anchor_for_large_volume_conditioning(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, _ = predictor.predict(
                PredictOptions(num_phases=2),
                anchors=[anchor],
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertTrue(torch.equal(volume[1], torch.ones(4, 4, dtype=torch.uint8)))
        self.assertTrue(torch.equal(volume[0], torch.zeros(4, 4, dtype=torch.uint8)))

    def test_predict_scale_up_accepts_vae_size_anchor_and_larger_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2),
                anchors=[anchor],
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(stats["condition_start"], 1)
        self.assertTrue(
            torch.equal(volume[2, 1:3, 1:3], torch.ones(2, 2, dtype=torch.uint8))
        )

    def test_predict_refines_large_volume_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2, refine=RefineConfig(steps=1)),
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIsInstance(stats, dict)

    def test_predict_runs_sds_for_large_volume_anchor(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(
                    num_phases=2,
                    sds=SDSConfig(steps=1, slice_steps=1, weight=0.0),
                    anchor=AnchorConfig(weight=1.0),
                ),
                anchors=[anchor],
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("steps", stats)

    def test_predict_scale_sds_visits_shifted_vae_size_anchor_slice(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(
                    num_phases=2,
                    sds=SDSConfig(
                        steps=1,
                        slice_steps=1,
                        batch_size=2,
                        weight=0.0,
                    ),
                    anchor=AnchorConfig(weight=1.0),
                ),
                anchors=[anchor],
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("anchor", stats)
        self.assertEqual(stats["volume_size"], 4)
        self.assertEqual(stats["latent_size"], 4)
        self.assertEqual(stats["tile_size"], 2)
        self.assertEqual(stats["condition_start"], 1)

    def test_predict_runs_scale_sds_with_large_target_images(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        options = PredictOptions(
            num_phases=2,
            sds=SDSConfig(steps=1, slice_steps=1, t_min=1, t_max=3, weight=0.0),
            targets=TargetConfig(vf_weight=1.0, tpc_weight=1.0),
        )
        target_images = [np.zeros((4, 4), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                options,
                target_images=target_images,
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("vf", stats)
        self.assertIn("tpc", stats)

    def test_predict_runs_scale_sds_with_vae_size_tpc_targets(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        options = PredictOptions(
            num_phases=2,
            sds=SDSConfig(steps=1, slice_steps=1, t_min=1, t_max=3, weight=0.0),
            targets=TargetConfig(tpc_weight=1.0),
        )
        target_images = [np.zeros((2, 2), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                options,
                target_images=target_images,
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("tpc", stats)


if __name__ == "__main__":
    unittest.main()
