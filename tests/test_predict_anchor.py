import unittest

import numpy as np
import torch

from src.predict import AnchorSlice
from src.predict.anchor import prepare_anchor_image, validate_anchor, validate_anchors
from src.predict.anchor.latent import prepare_anchor_latents


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)


class DownsamplingVAE(torch.nn.Module):
    image_size = 4
    latent_size = 2
    latent_ch = 1
    downsample_factor = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.nn.functional.avg_pool2d(image, kernel_size=2)
        return latent, torch.zeros_like(latent)


class PredictAnchorTest(unittest.TestCase):
    def test_prepare_anchor_image_scales_phase_image_to_tensor(self):
        image = np.array([[0, 1, 2]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 3]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(tensor[0, 0, 0], torch.tensor([-1.0, 0.0, 1.0]))
        )

    def test_prepare_anchor_image_can_segment_grayscale_image(self):
        image = np.array([[0, 0, 120, 120, 255, 255]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3, segment=True)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 6]))
        self.assertTrue(torch.equal(torch.unique(tensor), torch.tensor([-1.0, 0.0, 1.0])))

    def test_prepare_anchor_image_rejects_non_2d_image(self):
        image = np.zeros((1, 4, 4), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "2D"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_invalid_phase_values(self):
        image = np.array([[0, 2]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "0 to 1"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_num_phases_above_uint8_range(self):
        image = np.array([[0, 255]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "num_phases"):
            prepare_anchor_image(image, num_phases=257)

    def test_validate_anchor_accepts_matching_slice_shape(self):
        anchor = AnchorSlice(image=np.zeros((5, 6), dtype=np.uint8), axis=0, index=2)

        validate_anchor(anchor, volume_shape=(4, 5, 6))

    def test_validate_anchor_rejects_shape_mismatch(self):
        anchor = AnchorSlice(image=np.zeros((4, 6), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_anchor(anchor, volume_shape=(4, 5, 6))

    def test_validate_anchors_rejects_duplicate_axis_index(self):
        anchors = [
            AnchorSlice(image=np.zeros((5, 6), dtype=np.uint8), axis=0, index=2),
            AnchorSlice(image=np.ones((5, 6), dtype=np.uint8), axis=0, index=2),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_anchors(anchors, volume_shape=(4, 5, 6))

    def test_prepare_anchor_latents_maps_axis_zero(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=1)

        latent, mask = prepare_anchor_latents(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[1], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(mask[1], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(mask[0], torch.zeros(1, 2, 2)))

    def test_prepare_anchor_latents_rejects_same_latent_plane_collision(self):
        anchors = [
            AnchorSlice(image=np.zeros((4, 4), dtype=np.uint8), axis=0, index=0),
            AnchorSlice(image=np.ones((4, 4), dtype=np.uint8), axis=0, index=1),
        ]

        with self.assertRaisesRegex(ValueError, "latent plane"):
            prepare_anchor_latents(
                DownsamplingVAE(),
                anchors,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_prepare_anchor_latents_maps_axis_one(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=1, index=1)

        latent, mask = prepare_anchor_latents(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[:, :, 1, :], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, 1, :], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, 0, :], torch.zeros(2, 1, 2)))

    def test_prepare_anchor_latents_maps_axis_two(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=2, index=1)

        latent, mask = prepare_anchor_latents(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[:, :, :, 1], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, :, 1], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, :, 0], torch.zeros(2, 1, 2)))


if __name__ == "__main__":
    unittest.main()
