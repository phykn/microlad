import unittest

import numpy as np
import torch

from src.predict.scale.condition import (
    center_start,
    prepare_scale_anchor_latents,
    shifted_anchor_slices,
)
from src.predict.types import AnchorSlice


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)


class DownsamplingVAE(torch.nn.Module):
    image_size = 2
    latent_size = 1
    latent_ch = 1
    downsample_factor = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.mean(dim=(-2, -1), keepdim=True), torch.zeros(
            image.shape[0],
            1,
            1,
            1,
            dtype=image.dtype,
            device=image.device,
        )


class ScaleConditionTest(unittest.TestCase):
    def test_center_start_places_base_in_middle(self):
        self.assertEqual(center_start(volume_size=192, base_size=64), 64)
        self.assertEqual(center_start(volume_size=128, base_size=64), 32)

    def test_center_start_rejects_smaller_volume(self):
        with self.assertRaisesRegex(ValueError, "volume_size"):
            center_start(volume_size=32, base_size=64)

    def test_anchor_latent_is_written_at_shifted_center_plane(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=1)

        latent, mask = prepare_scale_anchor_latents(
            IdentityVAE(),
            [anchor],
            volume_size=6,
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertEqual(latent.shape, torch.Size([1, 6, 6, 6]))
        self.assertTrue(torch.equal(mask[:, 3, 2:4, 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(latent[:, 3, 2:4, 2:4], torch.ones(1, 2, 2)))

    def test_anchor_latents_reject_same_latent_plane_collision(self):
        anchors = [
            AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=2),
            AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=3),
        ]

        with self.assertRaisesRegex(ValueError, "latent plane"):
            prepare_scale_anchor_latents(
                DownsamplingVAE(),
                anchors,
                volume_size=8,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_shifted_anchor_slices_move_base_index_to_output_index(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=1)

        shifted = shifted_anchor_slices([anchor], volume_size=6, base_size=2)

        self.assertEqual(shifted, [(0, 3)])


if __name__ == "__main__":
    unittest.main()
