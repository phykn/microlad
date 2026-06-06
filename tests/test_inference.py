import unittest

import torch
import torch.nn as nn

from inference import (
    insert_condition_slice,
    p_sample_conditioned_slice,
    predict,
    predict_conditioned_volume,
    sample_conditioned_latent_volume,
    voxel_to_latent_index,
)
from models import DDPM


class ZeroSliceUNet(nn.Module):
    def forward(self, z_t, t, condition_z, axis, slice_index):
        return torch.zeros_like(z_t)


class DecodeVAE(nn.Module):
    def decode(self, z):
        return z[:, :1]


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

    def test_predict_conditioned_volume_returns_fixed_condition_error(self):
        vae = DecodeVAE()
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
        self.assertEqual(result["volume"].shape, torch.Size([16, 1, 16, 16]))
        self.assertEqual(result["latent_index"], 3)
        self.assertEqual(result["condition_error"], 0.0)

    def test_predict_alias_uses_same_conditioned_entrypoint(self):
        vae = DecodeVAE()
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


if __name__ == "__main__":
    unittest.main()
