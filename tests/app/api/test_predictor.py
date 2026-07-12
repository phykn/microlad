import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.app.api import AnchorSlice, PredictOptions, Predictor


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


class PredictOptionsTest(unittest.TestCase):
    def test_predict_options_rejects_non_integer_num_phases(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            PredictOptions(num_phases=2.5)

    def test_predict_options_rejects_num_phases_that_exceed_uint8_range(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            PredictOptions(num_phases=257)

    def test_predict_options_rejects_weights_outside_zero_to_one(self):
        with self.assertRaisesRegex(ValueError, "sds_weight"):
            PredictOptions(num_phases=2, sds_weight=1.1)
        with self.assertRaisesRegex(ValueError, "anchor_weight"):
            PredictOptions(num_phases=2, anchor_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "vf_weight"):
            PredictOptions(num_phases=2, vf_weight=2.0)

    def test_predict_options_rejects_non_finite_numeric_values(self):
        cases = [
            ("sds_weight", {"sds_weight": float("nan")}),
            ("anchor_weight", {"anchor_weight": float("nan")}),
            ("diffusivity_low_cond", {"diffusivity_low_cond": float("nan")}),
            ("sds_lr", {"sds_lr": float("nan")}),
        ]

        for message, kwargs in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    PredictOptions(num_phases=2, **kwargs)

    def test_predict_options_rejects_invalid_sds_batch_size(self):
        with self.assertRaisesRegex(ValueError, "sds_batch_size"):
            PredictOptions(num_phases=2, sds_batch_size=0)
        with self.assertRaisesRegex(ValueError, "sds_batch_size"):
            PredictOptions(num_phases=2, sds_batch_size=1.5)

    def test_predict_options_rejects_non_boolean_balanced_slices(self):
        with self.assertRaisesRegex(ValueError, "sds_balanced_slices"):
            PredictOptions(num_phases=2, sds_balanced_slices=1)
        with self.assertRaisesRegex(ValueError, "sds_consensus"):
            PredictOptions(num_phases=2, sds_consensus=1)

    def test_predict_options_rejects_non_integer_step_counts(self):
        cases = [
            ("sds_steps", {"sds_steps": 1.5}),
            ("sds_slice_steps", {"sds_slice_steps": True}),
            ("sds_t_min", {"sds_t_min": 1.5}),
            ("sds_t_max", {"sds_t_max": 2.5}),
            ("refine_steps", {"refine_steps": 1.5}),
            ("anchor_fit_steps", {"anchor_fit_steps": 1.5}),
            ("anchor_slab_radius", {"anchor_slab_radius": 1.5}),
            ("joint_3d_steps", {"joint_3d_steps": 1.5}),
            ("joint_3d_batch_size", {"joint_3d_batch_size": 1.5}),
            ("slicegan_steps", {"slicegan_steps": 1.5}),
            ("slicegan_hybrid_steps", {"slicegan_hybrid_steps": 1.5}),
            ("slicegan_condition_steps", {"slicegan_condition_steps": 1.5}),
            ("slicegan_finetune_steps", {"slicegan_finetune_steps": 1.5}),
            ("slicegan_seed", {"slicegan_seed": 1.5}),
        ]

        for message, kwargs in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, f"{message}.*integer"):
                    PredictOptions(num_phases=2, **kwargs)

    def test_predict_options_rejects_invalid_anchor_fit_lr(self):
        with self.assertRaisesRegex(ValueError, "anchor_fit_lr"):
            PredictOptions(num_phases=2, anchor_fit_lr=0.0)

    def test_predict_options_rejects_invalid_anchor_slab_values(self):
        with self.assertRaisesRegex(ValueError, "anchor_slab_radius"):
            PredictOptions(num_phases=2, anchor_slab_radius=-1)
        with self.assertRaisesRegex(ValueError, "anchor_slab_weight"):
            PredictOptions(num_phases=2, anchor_slab_weight=1.1)

    def test_predict_options_rejects_invalid_joint_3d_values(self):
        with self.assertRaisesRegex(ValueError, "lmpdd_axis_consensus"):
            PredictOptions(num_phases=2, lmpdd_axis_consensus=1)
        with self.assertRaisesRegex(ValueError, "anchor_latent_sigma"):
            PredictOptions(num_phases=2, anchor_latent_sigma=-1.0)
        with self.assertRaisesRegex(ValueError, "anchor_latent_strength"):
            PredictOptions(num_phases=2, anchor_latent_strength=0.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_batch_size"):
            PredictOptions(num_phases=2, joint_3d_batch_size=0)
        with self.assertRaisesRegex(ValueError, "joint_3d_lr"):
            PredictOptions(num_phases=2, joint_3d_lr=0.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_entropy_weight"):
            PredictOptions(num_phases=2, joint_3d_entropy_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_transition_weight"):
            PredictOptions(num_phases=2, joint_3d_transition_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_run_weight"):
            PredictOptions(num_phases=2, joint_3d_run_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_patch_weight"):
            PredictOptions(num_phases=2, joint_3d_patch_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_texture_weight"):
            PredictOptions(num_phases=2, joint_3d_texture_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_interface_weight"):
            PredictOptions(num_phases=2, joint_3d_interface_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "joint_3d_discriminator_lr"):
            PredictOptions(num_phases=2, joint_3d_discriminator_lr=0.0)
        with self.assertRaisesRegex(ValueError, "cannot both"):
            PredictOptions(num_phases=2, joint_3d_steps=1, sds_steps=1)
        with self.assertRaisesRegex(ValueError, "replaces refine_steps"):
            PredictOptions(num_phases=2, joint_3d_steps=1, refine_steps=1)
        with self.assertRaisesRegex(ValueError, "conditional SliceGAN replaces"):
            PredictOptions(num_phases=2, slicegan_steps=1, joint_3d_steps=1)

    def test_predict_options_rejects_negative_slicegan_values(self):
        for name in (
            "slicegan_steps",
            "slicegan_hybrid_steps",
            "slicegan_condition_steps",
            "slicegan_finetune_steps",
            "slicegan_seed",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    PredictOptions(num_phases=2, **{name: -1})


class PredictorTest(unittest.TestCase):
    def test_balanced_schedule_visits_every_slice_once_per_sweep(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")

        schedule = predictor._build_balanced_schedule(
            steps=6,
            batch_size=1,
            volume_size=2,
        )

        self.assertEqual(len(schedule), 6)
        self.assertEqual(set(schedule), {(axis, index) for axis in range(3) for index in range(2)})

    def test_predict_returns_quantized_phase_volume(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options=options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_predict_accepts_options_as_first_argument(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIsInstance(stats, dict)

    def test_predict_builds_targets_and_runs_sds_when_enabled(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(
            num_phases=2,
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            vf_weight=1.0,
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
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2, joint_3d_steps=1)
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
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2, slicegan_steps=1)
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

    def test_joint_prediction_applies_anchor_only_in_image_space(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
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
                options=PredictOptions(num_phases=2, joint_3d_steps=1),
            )

        self.assertIsNone(encode.call_args.args[1])

    def test_predict_blends_anchor_latent_without_forced_overwrite(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
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
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
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
                    refine_steps=1,
                    anchor_fit_steps=2,
                    anchor_fit_lr=0.1,
                ),
            )

        self.assertTrue(torch.equal(volume[1], torch.ones(2, 2, dtype=torch.uint8)))
        self.assertIn("anchor_fit", stats)

    def test_predict_requires_anchor_when_anchor_fit_is_enabled(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            with self.assertRaisesRegex(ValueError, "anchors"):
                predictor.predict(
                    PredictOptions(num_phases=2, anchor_fit_steps=1),
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
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2, sds_steps=1, vf_weight=1.0)

        with self.assertRaisesRegex(ValueError, "target_images"):
            predictor.predict(options)

    def test_predict_rejects_small_volume_target_images_with_wrong_size(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(
            num_phases=2,
            sds_steps=1,
            sds_weight=0.0,
            sa_weight=1.0,
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
                    sds_steps=1,
                    sds_slice_steps=1,
                    sds_t_min=1,
                    sds_t_max=4,
                    sds_weight=0.0,
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
        self.assertTrue(torch.equal(volume[2, 1:3, 1:3], torch.ones(2, 2, dtype=torch.uint8)))

    def test_predict_refines_large_volume_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2, refine_steps=1),
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
                    sds_steps=1,
                    sds_slice_steps=1,
                    sds_weight=0.0,
                    anchor_weight=1.0,
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
                    sds_steps=1,
                    sds_slice_steps=1,
                    sds_batch_size=2,
                    sds_weight=0.0,
                    anchor_weight=1.0,
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
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            vf_weight=1.0,
            tpc_weight=1.0,
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
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            tpc_weight=1.0,
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
