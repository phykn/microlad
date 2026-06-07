import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import build_grayscale_tpc_target
from src.inference import GenerationOptions, MicroLadPredictor
from src.inference.conditions import ConditionLock, FixedSlice
from src.inference.locked_sampling import (
    apply_condition_locks,
    denoise_axis,
    insert_condition_slice,
    sample_locked_latent_volume,
)
from src.inference.decoding import multi_axis_decode, three_axis_refinement
from src.inference.geometry import voxel_to_latent_index
from src.inference.sds import sds_refine_slice, sds_refine_volume
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
    def test_voxel_slice_index_maps_to_vae_latent_plane(self):
        self.assertEqual(voxel_to_latent_index(0), 0)
        self.assertEqual(voxel_to_latent_index(3), 0)
        self.assertEqual(voxel_to_latent_index(4), 1)
        self.assertEqual(voxel_to_latent_index(12), 3)

    def test_insert_condition_slice_sets_axis_position_in_latent_volume(self):
        volume = torch.zeros(4, 16, 16, 16)
        condition_z = torch.ones(4, 16, 16)

        result = insert_condition_slice(volume, condition_z, axis=0, slice_index=12)

        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, 2, :, :].sum()), 0.0)

    def test_insert_condition_slice_supports_all_axes(self):
        condition_z = torch.ones(4, 16, 16)

        axis_y = insert_condition_slice(torch.zeros(4, 16, 16, 16), condition_z, axis=1, slice_index=8)
        axis_x = insert_condition_slice(torch.zeros(4, 16, 16, 16), condition_z, axis=2, slice_index=8)

        self.assertEqual(float(axis_y[:, :, 2, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(axis_x[:, :, :, 2].sum()), 4 * 16 * 16)

    def test_insert_condition_slice_can_blend_lock_strength(self):
        volume = torch.zeros(4, 16, 16, 16)
        condition_z = torch.ones(4, 16, 16)

        result = insert_condition_slice(volume, condition_z, axis=0, slice_index=12, strength=0.25)

        self.assertTrue(torch.allclose(result[:, 3, :, :], torch.full((4, 16, 16), 0.25)))

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
        result = predictor.predict({"images": [{"image": condition, "axis": 0, "index": 12}]})

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
        result = predictor.predict({"images": [{"image": condition, "axis": 0, "index": 12}]})

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))

    def test_multi_axis_decode_returns_full_resolution_volume(self):
        vae = UpsampleVAE()
        volume_z = torch.zeros(4, 2, 2, 2)
        volume_z[0] = 1.0

        result = multi_axis_decode(vae, volume_z, downsample=4)

        self.assertEqual(result.shape, torch.Size([8, 1, 8, 8]))
        self.assertEqual(float(result.min()), 1.0)
        self.assertEqual(float(result.max()), 1.0)

    def test_three_axis_refinement_preserves_volume_shape(self):
        vae = UpsampleVAE()
        volume = torch.rand(8, 1, 8, 8)

        result = three_axis_refinement(volume, vae, refinement_steps=1)

        self.assertEqual(result.shape, volume.shape)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_sds_refine_slice_updates_selected_axis_slice(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)

        refined, losses = sds_refine_slice(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            axis=0,
            index=3,
            lr=0.1,
            t_min=1,
            t_max=3,
        )

        self.assertEqual(refined.shape, volume.shape)
        self.assertEqual(set(losses), {"loss", "sds", "vf", "tpc", "grayscale_tpc", "rd", "sa", "condition"})
        self.assertFalse(torch.allclose(refined[3], volume[3]))
        self.assertTrue(torch.allclose(refined[0], volume[0]))

    def test_sds_refine_slice_accepts_grayscale_tpc_target(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        volume = torch.full((8, 1, 8, 8), 0.5)
        target, bin_mat, bin_counts = build_grayscale_tpc_target(volume[3])

        refined, losses = sds_refine_slice(
            volume=volume,
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            axis=0,
            index=3,
            lr=0.1,
            t_min=1,
            t_max=3,
            grayscale_tpc_target=target,
            grayscale_tpc_bin_mat=bin_mat,
            grayscale_tpc_bin_counts=bin_counts,
            grayscale_tpc_weight=1.0,
        )

        self.assertEqual(refined.shape, volume.shape)
        self.assertIn("grayscale_tpc", losses)

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
        result = predictor.predict({"images": [{"image": condition, "axis": 0, "index": 12}]})

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
        result = predictor.predict({"images": [{"image": condition, "axis": 0, "index": 12}]})

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(len(result["sds_history"]), 1)

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
        result = predictor.predict({
            "images": [{"image": condition, "axis": 0, "index": 12}],
            "condition_weight": 1.0,
        })

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("condition", result["sds_history"][0])

    def test_predict_rejects_refinement_weights_without_sds_steps(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "sds_steps"):
            predictor.predict({
                "images": [{"image": condition, "axis": 0, "index": 12}],
                "condition_weight": 1.0,
            })
        with self.assertRaisesRegex(ValueError, "sds_steps"):
            predictor.predict({
                "images": [{"image": condition, "axis": 0, "index": 12}],
                "stats_weight": 1.0,
            })

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
        result = predictor.predict({
            "images": [{"image": condition, "axis": 0, "index": 12}],
            "stats_weight": 1.0,
        })

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("vf", result["sds_history"][0])
        self.assertIn("grayscale_tpc", result["sds_history"][0])
        self.assertIn("sa", result["sds_history"][0])

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
        result = predictor.predict({"images": [{"image": condition, "axis": 0, "index": 12}]})

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
        result = predictor.predict({
            "images": [
                {"image": condition, "axis": 0, "index": 12},
                {"image": condition, "axis": 1, "index": 8},
            ]
        })

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))
        self.assertTrue(torch.allclose(result["volume"][:, 0, 8, :], torch.ones(64, 64)))

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
        result = predictor.predict({
            "images": [
                {"image": condition, "axis": 0, "index": 12},
                {"image": condition, "axis": 1, "index": 8},
            ]
        })

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))

    def test_predict_resizes_condition_image_to_requested_size(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = np.full((32, 32), 255, dtype=np.float32)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict({
            "size": 128,
            "images": [{"image": condition, "axis": 0, "index": 32}],
        })

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))

    def test_predict_uses_condition_size_for_volume_shape(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict({"size": 128, "images": [{"image": condition, "axis": 0, "index": 32}]})

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))

    def test_predict_rejects_condition_index_outside_size(self):
        vae = UpsampleVAE()
        unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)
        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)

        with self.assertRaisesRegex(ValueError, "inside size"):
            predictor.predict({"size": 128, "images": [{"image": condition, "axis": 0, "index": 128}]})

    def test_predict_tiles_large_condition_crop(self):
        vae = UpsampleVAE()
        unet = RecordingTimeUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)

        predictor = MicroLadPredictor(vae=vae, unet=unet, ddpm=ddpm)
        result = predictor.predict({"size": 128, "images": [{"image": condition, "axis": 0, "index": 32}]})

        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))
        self.assertEqual(len(unet.calls), 27)
        self.assertEqual({call["shape"][-2:] for call in unet.calls}, {(16, 16)})


if __name__ == "__main__":
    unittest.main()
