import unittest

import numpy as np
import torch

from src.predict.blend import blend_window
from src.predict import AnchorSlice
from src.predict.anchor import (
    prepare_anchor_image,
    reconstruct_anchor_target,
    validate_anchor,
    validate_anchors,
)
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


class LocalPatternAnchorVAE(torch.nn.Module):
    image_size = 3

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image, torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        ).view(1, 1, self.image_size, 1)
        cols = torch.arange(
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        ).view(1, 1, 1, self.image_size)
        return (rows * 10.0 + cols).expand(latent.shape[0], 1, -1, -1)


class PredictAnchorTest(unittest.TestCase):
    def test_prepare_anchor_image_converts_phase_image_to_float_tensor(self):
        image = np.array([[0, 1, 2]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 3]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(tensor[0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
        )

    def test_prepare_anchor_image_can_segment_grayscale_image(self):
        image = np.array([[0, 0, 120, 120, 255, 255]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3, segment=True)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 6]))
        self.assertTrue(
            torch.equal(
                torch.unique(tensor),
                torch.tensor([0.0, 1.0, 2.0]),
            )
        )

    def test_prepare_anchor_image_rejects_non_2d_image(self):
        image = np.zeros((1, 4, 4), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "2D"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_invalid_phase_values(self):
        image = np.array([[0, 2]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "0 to 1"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_non_finite_phase_values(self):
        image = np.array([[np.nan]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_fractional_phase_values(self):
        image = np.array([[0.5]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "integer"):
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

    def test_prepare_anchor_latents_rejects_invalid_axis_directly(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=3,
            index=1,
        )

        with self.assertRaisesRegex(ValueError, "axis"):
            prepare_anchor_latents(
                IdentityVAE(),
                [anchor],
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_prepare_anchor_latents_rejects_invalid_index_directly(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=2,
        )

        with self.assertRaisesRegex(ValueError, "index"):
            prepare_anchor_latents(
                IdentityVAE(),
                [anchor],
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_reconstruct_anchor_target_uses_weighted_tile_blending(self):
        vae = LocalPatternAnchorVAE()
        image = torch.zeros(1, 1, 4, 4)

        recon = reconstruct_anchor_target(vae, image, tile_overlap=2)

        expected = torch.zeros_like(image)
        weight_sum = torch.zeros_like(image)
        window = blend_window(
            vae.image_size,
            vae.image_size,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        pattern = vae.decode(torch.zeros(1, 1, 3, 3))

        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            expected[:, :, row : row + 3, col : col + 3] += pattern * window
            weight_sum[:, :, row : row + 3, col : col + 3] += window

        expected = expected / weight_sum

        self.assertTrue(torch.allclose(recon, expected))


if __name__ == "__main__":
    unittest.main()
