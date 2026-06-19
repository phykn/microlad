import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference import GenerationOptions, MicroLadPredictor
from src.inference.condition_stats import build_condition_stats
from src.inference.conditions import ConditionLock, FixedSlice, condition_to_image
from src.inference.locked_sampling import (
    apply_condition_locks,
    denoise_axis,
    insert_condition_slice,
    sample_locked_latent_volume,
)
from src.inference.decoding import multi_axis_decode, three_axis_refinement
from src.inference.geometry import voxel_to_latent_index
from src.inference.sds import sds_refine_volume
from src.models import DDPM


class ZeroTimeUNet(nn.Module):
    def forward(self, z_t, t):
        return torch.zeros_like(z_t)


class RecordingTimeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, z_t, t):
        self.calls.append(
            {
                "shape": tuple(z_t.shape),
                "t": t.detach().cpu().clone(),
            }
        )
        return torch.zeros_like(z_t)


class UpsampleVAE(nn.Module):
    def encode(self, x):
        mu = F.avg_pool2d(x, kernel_size=4, stride=4).repeat(1, 4, 1, 1)
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return F.interpolate(z[:, :1], scale_factor=4, mode="nearest").clamp(0, 1)


class ConditionLockedInferenceTest(unittest.TestCase):
    def test_generation_options_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "volume_shape"):
            GenerationOptions(volume_shape=(4, 0, 16, 16))
        with self.assertRaisesRegex(ValueError, "refinement_steps"):
            GenerationOptions(refinement_steps=-1)
        with self.assertRaisesRegex(ValueError, "sds_steps"):
            GenerationOptions(sds_steps=-1)
        with self.assertRaisesRegex(ValueError, "sds_lr"):
            GenerationOptions(sds_lr=0.0)
        with self.assertRaisesRegex(ValueError, "t_min"):
            GenerationOptions(t_min=-1)
        with self.assertRaisesRegex(ValueError, "t_min"):
            GenerationOptions(t_min=10, t_max=10)
        with self.assertRaisesRegex(ValueError, "lock_strength"):
            GenerationOptions(lock_strength=-0.1)

    def test_voxel_slice_index_maps_to_vae_latent_plane(self):
        self.assertEqual(voxel_to_latent_index(0), 0)
        self.assertEqual(voxel_to_latent_index(3), 0)
        self.assertEqual(voxel_to_latent_index(4), 1)
        self.assertEqual(voxel_to_latent_index(12), 3)

    def test_voxel_slice_index_rejects_invalid_downsample(self):
        with self.assertRaisesRegex(ValueError, "downsample"):
            voxel_to_latent_index(12, downsample=0)

    def test_condition_to_image_normalizes_uint16_to_unit_range(self):
        condition = np.linspace(0, 4095, num=64 * 64, dtype=np.uint16).reshape(64, 64)

        image = condition_to_image(condition, torch.device("cpu"))

        self.assertEqual(image.shape, torch.Size([1, 1, 64, 64]))
        self.assertEqual(float(image.min()), 0.0)
        self.assertEqual(float(image.max()), 1.0)

    def test_condition_to_image_converts_rgb_to_gray(self):
        condition = np.zeros((64, 64, 3), dtype=np.uint8)
        condition[..., 0] = 255

        image = condition_to_image(condition, torch.device("cpu"))

        self.assertEqual(image.shape, torch.Size([1, 1, 64, 64]))
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 0.299, places=3)

    def test_insert_condition_slice_sets_axis_position_in_latent_volume(self):
        volume = torch.zeros(4, 16, 16, 16)
        condition_z = torch.ones(4, 16, 16)

        result = insert_condition_slice(volume, condition_z, axis=0, slice_index=12)

        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, 2, :, :].sum()), 0.0)

    def test_insert_condition_slice_supports_all_axes(self):
        condition_z = torch.ones(4, 16, 16)

        axis_y = insert_condition_slice(
            torch.zeros(4, 16, 16, 16), condition_z, axis=1, slice_index=8
        )
        axis_x = insert_condition_slice(
            torch.zeros(4, 16, 16, 16), condition_z, axis=2, slice_index=8
        )

        self.assertEqual(float(axis_y[:, :, 2, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(axis_x[:, :, :, 2].sum()), 4 * 16 * 16)

    def test_insert_condition_slice_can_blend_lock_strength(self):
        volume = torch.zeros(4, 16, 16, 16)
        condition_z = torch.ones(4, 16, 16)

        result = insert_condition_slice(
            volume, condition_z, axis=0, slice_index=12, strength=0.25
        )

        self.assertTrue(
            torch.allclose(result[:, 3, :, :], torch.full((4, 16, 16), 0.25))
        )

    def test_insert_condition_slice_rejects_shape_and_position_mismatch(self):
        volume = torch.zeros(4, 2, 4, 4)

        with self.assertRaisesRegex(ValueError, "channel"):
            insert_condition_slice(volume, torch.ones(3, 4, 4), axis=0, slice_index=0)
        with self.assertRaisesRegex(ValueError, "inside volume_z"):
            insert_condition_slice(volume, torch.ones(4, 4, 4), axis=0, slice_index=8)
        with self.assertRaisesRegex(ValueError, "inside volume_z"):
            insert_condition_slice(
                volume,
                torch.ones(4, 4, 4),
                axis=0,
                slice_index=0,
                row_start=1,
            )

    def test_apply_condition_locks_sets_multiple_positions(self):
        condition_z = torch.ones(4, 16, 16)

        result = apply_condition_locks(
            torch.zeros(4, 16, 16, 16),
            [
                ConditionLock(condition_z=condition_z, axis=0, slice_index=12),
                ConditionLock(condition_z=condition_z, axis=1, slice_index=8),
            ],
        )

        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, :, 2, :].sum()), 4 * 16 * 16)

    def test_denoise_axis_preserves_volume_shape(self):
        volume = torch.zeros(4, 16, 16, 16)
        t = torch.tensor([0])

        result = denoise_axis(
            ZeroTimeUNet(),
            DDPM(timesteps=10),
            volume,
            t,
            axis=0,
        )

        self.assertEqual(result.shape, volume.shape)

    def test_denoise_axis_rejects_invalid_axis(self):
        volume = torch.zeros(4, 16, 16, 16)
        t = torch.tensor([0])

        with self.assertRaisesRegex(ValueError, "axis"):
            denoise_axis(
                ZeroTimeUNet(),
                DDPM(timesteps=10),
                volume,
                t,
                axis=3,
            )

    def test_locked_sampling_loop_keeps_condition_slice(self):
        condition_z = torch.ones(4, 16, 16)

        result = sample_locked_latent_volume(
            ZeroTimeUNet(),
            DDPM(timesteps=3),
            [ConditionLock(condition_z=condition_z, axis=0, slice_index=12)],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result.shape, torch.Size([4, 16, 16, 16]))
        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)

    def test_locked_sampling_loop_accepts_soft_lock_strength(self):
        condition_z = torch.ones(4, 16, 16)

        result = sample_locked_latent_volume(
            ZeroTimeUNet(),
            DDPM(timesteps=1),
            [ConditionLock(condition_z=condition_z, axis=0, slice_index=12)],
            volume_shape=(4, 16, 16, 16),
            lock_strength=0.5,
        )

        self.assertEqual(result.shape, torch.Size([4, 16, 16, 16]))
        self.assertFalse(torch.allclose(result[:, 3, :, :], condition_z))

    def test_locked_sampling_loop_denoises_all_axes_each_step(self):
        condition_z = torch.ones(4, 16, 16)
        unet = RecordingTimeUNet()

        sample_locked_latent_volume(
            unet,
            DDPM(timesteps=1),
            [ConditionLock(condition_z=condition_z, axis=1, slice_index=8)],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual([call["shape"] for call in unet.calls], [(16, 4, 16, 16)] * 3)

    def test_locked_sampling_loop_keeps_all_slices(self):
        condition_z = torch.ones(4, 16, 16)

        result = sample_locked_latent_volume(
            ZeroTimeUNet(),
            DDPM(timesteps=3),
            [
                ConditionLock(condition_z=condition_z, axis=0, slice_index=12),
                ConditionLock(condition_z=condition_z, axis=1, slice_index=8),
            ],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result.shape, torch.Size([4, 16, 16, 16]))
        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, :, 2, :].sum()), 4 * 16 * 16)

    def test_locked_sampling_loop_uses_one_unet_call_per_axis(self):
        condition_z = torch.ones(4, 16, 16)
        unet = RecordingTimeUNet()

        sample_locked_latent_volume(
            unet,
            DDPM(timesteps=1),
            [ConditionLock(condition_z=condition_z, axis=0, slice_index=12)],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual([call["shape"] for call in unet.calls], [(16, 4, 16, 16)] * 3)

    def test_predict_returns_conditioned_volume(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16)),
        )
        result = predictor.predict(
            {"images": [{"image": condition, "axis": 0, "index": 12}]}
        )

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(set(result), {"volume", "sds_history"})

    def test_predictor_class_can_generate_without_conditions(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16)),
        )
        result = predictor.predict()

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(result["sds_history"], [])

    def test_predict_can_generate_requested_size_without_images(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict({"size": 128})

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))
        self.assertEqual(result["sds_history"], [])

    def test_predictor_class_uses_stored_generation_options(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16), lock_strength=1.0),
        )
        result = predictor.predict(
            {"images": [{"image": condition, "axis": 0, "index": 12}]}
        )

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))

    def test_multi_axis_decode_returns_full_resolution_volume(self):
        vae = UpsampleVAE()
        volume_z = torch.zeros(4, 2, 2, 2)
        volume_z[0] = 1.0

        result = multi_axis_decode(vae, volume_z, downsample=4)

        self.assertEqual(result.shape, torch.Size([8, 1, 8, 8]))
        self.assertEqual(float(result.min()), 1.0)
        self.assertEqual(float(result.max()), 1.0)

    def test_multi_axis_decode_rejects_invalid_downsample(self):
        vae = UpsampleVAE()
        volume_z = torch.zeros(4, 2, 2, 2)

        with self.assertRaisesRegex(ValueError, "downsample"):
            multi_axis_decode(vae, volume_z, downsample=0)

    def test_three_axis_refinement_preserves_volume_shape(self):
        vae = UpsampleVAE()
        volume = torch.rand(8, 1, 8, 8)

        result = three_axis_refinement(volume, vae, refinement_steps=1)

        self.assertEqual(result.shape, volume.shape)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_three_axis_refinement_rejects_multichannel_volume(self):
        vae = UpsampleVAE()
        volume = torch.rand(8, 2, 8, 8)

        with self.assertRaisesRegex(ValueError, "single gray channel"):
            three_axis_refinement(volume, vae, refinement_steps=1)

    def test_sds_refine_volume_rejects_invalid_inputs(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)

        with self.assertRaisesRegex(ValueError, "single gray channel"):
            sds_refine_volume(
                volume=torch.full((8, 2, 8, 8), 0.5),
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=1,
                lr=0.1,
                t_min=1,
                t_max=3,
            )
        with self.assertRaisesRegex(ValueError, "lr"):
            sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=1,
                lr=0.0,
                t_min=1,
                t_max=3,
            )
        with self.assertRaisesRegex(ValueError, "t_min"):
            sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=1,
                lr=0.1,
                t_min=3,
                t_max=3,
            )

    def test_sds_refine_volume_computes_stats_targets_from_condition_images(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)
        condition = torch.ones(1, 1, 8, 8)

        refined, history = sds_refine_volume(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            condition_images=[condition],
            stats_weight=1.0,
        )

        self.assertEqual(refined.shape, volume.shape)
        self.assertIn("gray_moment", history[0])
        self.assertIn("grayscale_tpc", history[0])
        self.assertIn("surface_area", history[0])
        self.assertNotIn("tpc", history[0])

    def test_sds_refine_volume_runs_multiple_steps_and_final_refinement(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)

        refined, history = sds_refine_volume(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            steps=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            refinement_steps=1,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertEqual(refined.shape, volume.shape)
        self.assertEqual(len(history), 2)
        self.assertIn("loss", history[0])
        self.assertFalse(torch.allclose(refined, volume))

    def test_sds_refine_volume_reinserts_fixed_slices(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)
        fixed = torch.ones(8, 8)

        refined, _ = sds_refine_volume(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            steps=2,
            lr=0.1,
            t_min=1,
            t_max=3,
            fixed_slices=[FixedSlice(axis=0, index=3, image=fixed)],
            generator=torch.Generator().manual_seed(0),
        )

        self.assertTrue(torch.allclose(refined[3, 0], fixed))

    def test_sds_refine_volume_rejects_invalid_fixed_slice(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)

        with self.assertRaisesRegex(ValueError, "slice index"):
            sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=0,
                lr=0.1,
                t_min=1,
                t_max=3,
                fixed_slices=[FixedSlice(axis=0, index=-1, image=torch.ones(8, 8))],
            )
        with self.assertRaisesRegex(ValueError, "slice image shape"):
            sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=0,
                lr=0.1,
                t_min=1,
                t_max=3,
                fixed_slices=[FixedSlice(axis=0, index=3, image=torch.ones(4, 8))],
            )

    def test_sds_refine_volume_applies_condition_loss_to_condition_slice(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.75)
        target = torch.ones(8, 8)

        refined, history = sds_refine_volume(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            steps=1,
            lr=0.1,
            t_min=1,
            t_max=3,
            condition_slices=[FixedSlice(axis=0, index=3, image=target)],
            condition_weight=100.0,
        )

        self.assertEqual(len(history), 1)
        self.assertIn("condition", history[0])
        initial_mu, _ = vae.encode(volume[3].unsqueeze(0) * 2 - 1)
        initial_decoded = vae.decode(initial_mu)[0, 0]
        self.assertLess(
            float(F.mse_loss(refined[3, 0], target)),
            float(F.mse_loss(initial_decoded, target)),
        )

    def test_sds_refine_volume_rejects_condition_weight_without_condition_slice(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)

        with self.assertRaisesRegex(ValueError, "condition_slices"):
            sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=1,
                lr=0.1,
                t_min=1,
                t_max=3,
                condition_weight=1.0,
            )

    def test_predict_accepts_numpy_condition_image(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = np.full((64, 64), 255, dtype=np.float32)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16)),
        )
        result = predictor.predict(
            {"images": [{"image": condition, "axis": 0, "index": 12}]}
        )

        self.assertEqual(set(result), {"volume", "sds_history"})

    def test_predict_can_apply_sds_with_separate_unet(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                volume_shape=(4, 16, 16, 16),
                sds_steps=1,
                sds_unet=sds_unet,
                t_min=1,
                t_max=3,
            ),
        )
        result = predictor.predict(
            {"images": [{"image": condition, "axis": 0, "index": 12}]}
        )

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(len(result["sds_history"]), 1)

    def test_predict_passes_refinement_to_sds_without_pre_refining(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.zeros(8, 1, 8, 8)
        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                refinement_steps=2,
                sds_steps=1,
                t_min=1,
                t_max=3,
            ),
        )

        with (
            patch("src.inference.predict.three_axis_refinement") as refinement,
            patch(
                "src.inference.predict.sds_refine_volume",
                return_value=(volume, []),
            ) as sds,
        ):
            predictor._refine_decoded_volume(
                volume=volume,
                condition_images=None,
                condition_slices=None,
                condition_weight=0.0,
                stats_weight=0.0,
                diffusivity_weight=0.0,
                diffusivity_size=32,
                fixed_slices=None,
            )

        refinement.assert_not_called()
        self.assertEqual(sds.call_args.kwargs["refinement_steps"], 2)

    def test_predict_passes_condition_weight_to_sds_refinement(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                volume_shape=(4, 16, 16, 16),
                sds_steps=1,
                sds_unet=sds_unet,
                t_min=1,
                t_max=3,
                lock_condition_slice=False,
            ),
        )
        result = predictor.predict(
            {
                "images": [{"image": condition, "axis": 0, "index": 12}],
                "condition_weight": 1.0,
            }
        )

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("condition", result["sds_history"][0])

    def test_predict_rejects_refinement_weights_without_sds_steps(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "sds_steps"):
            predictor.predict(
                {
                    "images": [{"image": condition, "axis": 0, "index": 12}],
                    "condition_weight": 1.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "sds_steps"):
            predictor.predict(
                {
                    "images": [{"image": condition, "axis": 0, "index": 12}],
                    "stats_weight": 1.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "sds_steps"):
            predictor.predict(
                {
                    "images": [{"image": condition, "axis": 0, "index": 12}],
                    "diffusivity_weight": 1.0,
                }
            )

    def test_predict_can_build_condition_stats_from_condition_image(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.rand(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                volume_shape=(4, 16, 16, 16),
                sds_steps=1,
                sds_unet=sds_unet,
                t_min=1,
                t_max=3,
            ),
        )
        result = predictor.predict(
            {
                "images": [{"image": condition, "axis": 0, "index": 12}],
                "stats_weight": 1.0,
            }
        )

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("gray_moment", result["sds_history"][0])
        self.assertIn("grayscale_tpc", result["sds_history"][0])
        self.assertIn("surface_area", result["sds_history"][0])

    def test_predict_can_build_diffusivity_targets_from_condition_image(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.rand(1, 1, 8, 8)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                volume_shape=(4, 16, 16, 16),
                sds_steps=1,
                sds_unet=sds_unet,
                t_min=1,
                t_max=3,
            ),
        )
        result = predictor.predict(
            {
                "size": 8,
                "images": [{"image": condition, "axis": 0, "index": 4}],
                "diffusivity_weight": 1.0,
            }
        )

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("diffusivity", result["sds_history"][0])

    def test_condition_stats_can_limit_diffusivity_solver_size(self):
        image = torch.rand(1, 1, 8, 8)

        stats = build_condition_stats(
            condition_images=[image],
            stats_weight=0.0,
            diffusivity_weight=1.0,
            diffusivity_size=4,
            gray_levels=[0, 1],
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(stats.diffusivity_targets)
        self.assertEqual(stats.diffusivity_solver.height, 4)
        self.assertEqual(stats.diffusivity_solver.width, 4)

    def test_condition_stats_rejects_negative_weights(self):
        image = torch.rand(1, 1, 8, 8)

        with self.assertRaisesRegex(ValueError, "stats_weight"):
            build_condition_stats(
                condition_images=[image],
                stats_weight=-1.0,
                diffusivity_weight=0.0,
                diffusivity_size=4,
                gray_levels=[0, 1],
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "diffusivity_weight"):
            build_condition_stats(
                condition_images=[image],
                stats_weight=0.0,
                diffusivity_weight=-1.0,
                diffusivity_size=4,
                gray_levels=[0, 1],
                device=torch.device("cpu"),
            )

    def test_predict_locks_condition_slice_during_sds(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(
                volume_shape=(4, 16, 16, 16),
                sds_steps=1,
                sds_unet=sds_unet,
                t_min=1,
                t_max=3,
                lock_condition_slice=True,
            ),
        )
        result = predictor.predict(
            {"images": [{"image": condition, "axis": 0, "index": 12}]}
        )

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))

    def test_predict_locks_multiple_condition_slices(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16)),
        )
        result = predictor.predict(
            {
                "images": [
                    {"image": condition, "axis": 0, "index": 12},
                    {"image": condition, "axis": 1, "index": 8},
                ]
            }
        )

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))
        self.assertTrue(
            torch.allclose(result["volume"][:, 0, 8, :], torch.ones(64, 64))
        )

    def test_predict_accepts_condition_specs(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        predictor = MicroLadPredictor(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            options=GenerationOptions(volume_shape=(4, 16, 16, 16)),
        )
        result = predictor.predict(
            {
                "images": [
                    {"image": condition, "axis": 0, "index": 12},
                    {"image": condition, "axis": 1, "index": 8},
                ]
            }
        )

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))

    def test_predict_resizes_condition_image_to_requested_size(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = np.full((32, 32), 255, dtype=np.float32)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict(
            {
                "size": 128,
                "images": [{"image": condition, "axis": 0, "index": 32}],
            }
        )

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))

    def test_predict_uses_condition_size_for_volume_shape(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict(
            {"size": 128, "images": [{"image": condition, "axis": 0, "index": 32}]}
        )

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))

    def test_predict_rejects_condition_index_outside_size(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "inside size"):
            predictor.predict(
                {"size": 128, "images": [{"image": condition, "axis": 0, "index": 128}]}
            )

    def test_predict_rejects_condition_without_axis(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 64, 64)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "condition image axis"):
            predictor.predict({"images": [{"image": condition, "index": 12}]})

    def test_predict_rejects_condition_without_index(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 64, 64)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "condition image index"):
            predictor.predict({"images": [{"image": condition, "axis": 0}]})

    def test_predict_rejects_condition_axis_outside_xyz(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 64, 64)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "condition image axis"):
            predictor.predict(
                {"images": [{"image": condition, "axis": 3, "index": 12}]}
            )

    def test_predict_tiles_large_condition_crop(self):
        vae = UpsampleVAE()
        unet = RecordingTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict(
            {"size": 128, "images": [{"image": condition, "axis": 0, "index": 32}]}
        )

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))
        self.assertEqual(len(unet.calls), 27)
        self.assertEqual({call["shape"][-2:] for call in unet.calls}, {(16, 16)})


if __name__ == "__main__":
    unittest.main()
