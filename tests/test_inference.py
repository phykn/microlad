import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import build_grayscale_tpc_target
from inference import (
    ConditionSpec,
    PredictConfig,
    insert_condition_slice,
    insert_condition_slices,
    multi_axis_decode,
    p_sample_conditioned_slice,
    predict,
    predict_conditioned_volume,
    predict_many,
    predict_scale_up,
    predict_with_config,
    sample_conditioned_latent_volume,
    sample_conditioned_latent_volume_multi,
    sds_refine_slice,
    sds_refine_volume,
    three_axis_refinement,
    voxel_to_latent_index,
)
from models import DDPM


class ZeroSliceUNet(nn.Module):
    def forward(self, z_t, t, condition_z, axis, slice_index):
        return torch.zeros_like(z_t)


class ZeroTimeUNet(nn.Module):
    def forward(self, z_t, t):
        return torch.zeros_like(z_t)


class UpsampleVAE(nn.Module):
    def encode(self, x):
        mu = F.avg_pool2d(x, kernel_size=4, stride=4).repeat(1, 4, 1, 1)
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return F.interpolate(z[:, :1], scale_factor=4, mode="nearest").clamp(0, 1)


class SliceConditionedInferenceTest(unittest.TestCase):
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

    def test_insert_condition_slices_sets_multiple_positions(self):
        condition_z = torch.ones(4, 16, 16)

        result = insert_condition_slices(
            torch.zeros(4, 16, 16, 16),
            [
                {"condition_z": condition_z, "axis": 0, "slice_index": 12},
                {"condition_z": condition_z, "axis": 1, "slice_index": 8},
            ],
        )

        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, :, 2, :].sum()), 4 * 16 * 16)

    def test_conditioned_sample_step_reinserts_condition_slice(self):
        volume = torch.zeros(4, 16, 16, 16)
        condition_z = torch.ones(4, 16, 16)
        t = torch.tensor([0])

        result = p_sample_conditioned_slice(
            ZeroSliceUNet(),
            DDPM(timesteps=10),
            volume,
            t,
            condition_z,
            axis=0,
            slice_index=12,
        )

        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)

    def test_conditioned_sampling_loop_keeps_condition_slice(self):
        condition_z = torch.ones(4, 16, 16)

        result = sample_conditioned_latent_volume(
            ZeroSliceUNet(),
            DDPM(timesteps=3),
            condition_z,
            axis=0,
            slice_index=12,
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result.shape, torch.Size([4, 16, 16, 16]))
        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)

    def test_multi_conditioned_sampling_loop_keeps_all_slices(self):
        condition_z = torch.ones(4, 16, 16)

        result = sample_conditioned_latent_volume_multi(
            ZeroSliceUNet(),
            DDPM(timesteps=3),
            [
                {"condition_z": condition_z, "axis": 0, "slice_index": 12},
                {"condition_z": condition_z, "axis": 1, "slice_index": 8},
            ],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result.shape, torch.Size([4, 16, 16, 16]))
        self.assertEqual(float(result[:, 3, :, :].sum()), 4 * 16 * 16)
        self.assertEqual(float(result[:, :, 2, :].sum()), 4 * 16 * 16)

    def test_predict_conditioned_volume_returns_fixed_condition_error(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        ddpm = DDPM(timesteps=3)
        condition_z = torch.ones(1, 4, 16, 16)

        result = predict_conditioned_volume(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            condition=condition_z,
            axis=0,
            slice_index=12,
            condition_is_latent=True,
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result["volume_z"].shape, torch.Size([4, 16, 16, 16]))
        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(result["latent_index"], 3)
        self.assertEqual(result["condition_error"], 0.0)

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
        self.assertEqual(set(losses), {"loss", "sds", "vf", "tpc", "grayscale_tpc", "rd", "sa"})
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
            fixed_slices=[{"axis": 0, "index": 3, "image": fixed}],
            generator=torch.Generator().manual_seed(0),
        )

        self.assertTrue(torch.allclose(refined[3, 0], fixed))

    def test_predict_alias_uses_same_conditioned_entrypoint(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        ddpm = DDPM(timesteps=3)
        condition_z = torch.ones(1, 4, 16, 16)

        result = predict(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            condition=condition_z,
            axis=0,
            slice_index=12,
            condition_is_latent=True,
            volume_shape=(4, 16, 16, 16),
        )

        self.assertEqual(result["condition_error"], 0.0)

    def test_predict_can_apply_sds_with_separate_unet(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition_z = torch.ones(1, 4, 16, 16)

        result = predict(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            condition=condition_z,
            axis=0,
            slice_index=12,
            condition_is_latent=True,
            volume_shape=(4, 16, 16, 16),
            sds_steps=1,
            sds_unet=sds_unet,
            t_min=1,
            t_max=3,
        )

        self.assertEqual(result["volume"].shape, torch.Size([64, 1, 64, 64]))
        self.assertEqual(len(result["sds_history"]), 1)

    def test_predict_can_build_grayscale_tpc_from_condition_image(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.rand(1, 1, 64, 64)

        result = predict(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            condition=condition,
            axis=0,
            slice_index=12,
            volume_shape=(4, 16, 16, 16),
            sds_steps=1,
            sds_unet=sds_unet,
            t_min=1,
            t_max=3,
            use_condition_tpc=True,
            condition_tpc_weight=1.0,
        )

        self.assertEqual(len(result["sds_history"]), 1)
        self.assertIn("grayscale_tpc", result["sds_history"][0])

    def test_predict_locks_condition_slice_during_sds(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        sds_unet = ZeroTimeUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        result = predict(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            condition=condition,
            axis=0,
            slice_index=12,
            volume_shape=(4, 16, 16, 16),
            sds_steps=1,
            sds_unet=sds_unet,
            t_min=1,
            t_max=3,
            lock_condition_slice=True,
        )

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))

    def test_predict_many_locks_multiple_condition_slices(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)

        result = predict_many(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            conditions=[
                {"condition": condition, "axis": 0, "slice_index": 12},
                {"condition": condition, "axis": 1, "slice_index": 8},
            ],
            volume_shape=(4, 16, 16, 16),
        )

        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))
        self.assertTrue(torch.allclose(result["volume"][:, 0, 8, :], torch.ones(64, 64)))

    def test_predict_with_config_uses_condition_specs(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        ddpm = DDPM(timesteps=3)
        condition = torch.ones(1, 1, 64, 64)
        config = PredictConfig(
            conditions=[
                ConditionSpec(condition=condition, axis=0, slice_index=12),
                ConditionSpec(condition=condition, axis=1, slice_index=8),
            ],
            volume_shape=(4, 16, 16, 16),
        )

        result = predict_with_config(vae=vae, unet=unet, ddpm=ddpm, config=config)

        self.assertEqual(result["condition_errors"], [0.0, 0.0])
        self.assertTrue(torch.allclose(result["volume"][12], condition.squeeze(0)))

    def test_predict_scale_up_uses_crop_size_for_volume_shape(self):
        vae = UpsampleVAE()
        unet = ZeroSliceUNet()
        ddpm = DDPM(timesteps=1)
        condition = torch.ones(1, 1, 128, 128)

        result = predict_scale_up(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            conditions=[ConditionSpec(condition=condition, axis=0, slice_index=32)],
            output_size=128,
            latent_ch=4,
        )

        self.assertEqual(result["volume_z"].shape, torch.Size([4, 32, 32, 32]))
        self.assertEqual(result["volume"].shape, torch.Size([128, 1, 128, 128]))
        self.assertEqual(result["condition_errors"], [0.0])


if __name__ == "__main__":
    unittest.main()
