import unittest

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.pipeline.predict.guidance.conditioning.images import (
    prepare_anchor_image,
    prepare_volume_anchors,
)
from src.pipeline.predict.guidance.conditioning.latents import encode_anchors
from src.pipeline.predict.guidance.conditioning.reconstruction import reconstruct_target
from src.pipeline.predict.guidance.conditioning.validation import validate_anchor, validate_anchors
from src.pipeline.predict.scaling.tiles import blend_window


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
    num_phases = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image, torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
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
        phase_one = 0.1 + 0.8 * (rows + cols) / (2 * (self.image_size - 1))
        phase_one = phase_one.expand(latent.shape[0], 1, -1, -1)
        return torch.cat((1.0 - phase_one, phase_one), dim=1)


class ScalarVAE(torch.nn.Module):
    image_size = 2
    num_phases = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image, torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


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

    def test_validate_anchor_rejects_boolean_axis_and_index(self):
        image = np.zeros((5, 6), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "axis"):
            validate_anchor(AnchorSlice(image, axis=True, index=0), (4, 5, 6))
        with self.assertRaisesRegex(ValueError, "index"):
            validate_anchor(AnchorSlice(image, axis=0, index=True), (4, 5, 6))

    def test_validate_anchors_rejects_duplicate_axis_index(self):
        anchors = [
            AnchorSlice(image=np.zeros((5, 6), dtype=np.uint8), axis=0, index=2),
            AnchorSlice(image=np.ones((5, 6), dtype=np.uint8), axis=0, index=2),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_anchors(anchors, volume_shape=(4, 5, 6))

    def test_prepare_volume_anchors_rejects_cross_axis_conflicts(self):
        anchors = [
            AnchorSlice(np.zeros((2, 2), dtype=np.uint8), axis=0, index=1),
            AnchorSlice(
                np.array([[0, 0], [1, 1]], dtype=np.uint8),
                axis=1,
                index=0,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "Conflicting anchor intersection"):
            prepare_volume_anchors(
                anchors,
                volume_size=2,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_encode_anchors_maps_axis_zero(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=1)

        latent, mask = encode_anchors(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[1], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(mask[1], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(mask[0], torch.zeros(1, 2, 2)))

    def test_encode_anchors_can_spread_a_soft_condition_across_planes(self):
        torch.manual_seed(0)
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=0)

        latent, mask = encode_anchors(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
            spread_sigma=1.0,
            peak_strength=0.8,
        )

        self.assertTrue(torch.equal(latent[0], torch.ones(1, 2, 2)))
        self.assertFalse(torch.equal(latent[1], latent[0]))
        self.assertTrue(torch.allclose(mask[0], torch.full((1, 2, 2), 0.8)))
        expected_neighbor = 0.8 * np.exp(-0.5)
        self.assertTrue(
            torch.allclose(
                mask[1],
                torch.full((1, 2, 2), expected_neighbor),
            )
        )

    def test_encode_anchors_rejects_same_latent_plane_collision(self):
        anchors = [
            AnchorSlice(image=np.zeros((4, 4), dtype=np.uint8), axis=0, index=0),
            AnchorSlice(image=np.ones((4, 4), dtype=np.uint8), axis=0, index=1),
        ]

        with self.assertRaisesRegex(ValueError, "latent plane"):
            encode_anchors(
                DownsamplingVAE(),
                anchors,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_encode_anchors_maps_axis_one(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=1, index=1)

        latent, mask = encode_anchors(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[:, :, 1, :], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, 1, :], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, 0, :], torch.zeros(2, 1, 2)))

    def test_encode_anchors_maps_axis_two(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=2, index=1)

        latent, mask = encode_anchors(
            IdentityVAE(),
            [anchor],
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(latent[:, :, :, 1], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, :, 1], torch.ones(2, 1, 2)))
        self.assertTrue(torch.equal(mask[:, :, :, 0], torch.zeros(2, 1, 2)))

    def test_encode_anchors_rejects_invalid_axis_directly(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=3,
            index=1,
        )

        with self.assertRaisesRegex(ValueError, "axis"):
            encode_anchors(
                IdentityVAE(),
                [anchor],
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_encode_anchors_rejects_invalid_index_directly(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=2,
        )

        with self.assertRaisesRegex(ValueError, "index"):
            encode_anchors(
                IdentityVAE(),
                [anchor],
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_reconstruct_target_uses_weighted_tile_blending(self):
        vae = LocalPatternAnchorVAE()
        image = torch.zeros(1, 1, 4, 4)

        recon = reconstruct_target(vae, image, tile_overlap=2)

        expected = torch.zeros(1, 2, 4, 4)
        weight_sum = torch.zeros_like(image)
        window = blend_window(
            vae.image_size,
            vae.image_size,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        pattern = vae.decode_probs(torch.zeros(1, 1, 3, 3))

        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            expected[:, :, row : row + 3, col : col + 3] += pattern * window
            weight_sum[:, :, row : row + 3, col : col + 3] += window

        expected = expected / weight_sum
        expected = probabilities_to_calibrated_labels(expected, num_phases=2).float()

        self.assertTrue(torch.equal(recon, expected))

    def test_reconstruct_target_rejects_scalar_vae(self):
        with self.assertRaisesRegex(TypeError, "decode_probs"):
            reconstruct_target(ScalarVAE(), torch.zeros(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
