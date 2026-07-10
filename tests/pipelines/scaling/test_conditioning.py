import unittest

import numpy as np
import torch

from src.pipelines.scaling.conditioning import (
    center_start,
    encode_scale_anchors,
    build_scale_targets,
    shift_anchor_slices,
)
from src.app.api.options import AnchorSlice


class IdentityVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


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

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.expand(-1, 1, self.image_size, self.image_size)


class ShiftDecodeVAE(IdentityVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + 0.25


class NonFiniteVAE(IdentityVAE):
    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class ScaleConditionTest(unittest.TestCase):
    def test_center_start_places_base_in_middle(self):
        self.assertEqual(center_start(volume_size=192, base_size=64), 64)
        self.assertEqual(center_start(volume_size=128, base_size=64), 32)

    def test_center_start_rejects_smaller_volume(self):
        with self.assertRaisesRegex(ValueError, "volume_size"):
            center_start(volume_size=32, base_size=64)

    def test_anchor_latent_is_written_at_shifted_center_plane(self):
        anchor = AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=1)

        latent, mask = encode_scale_anchors(
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
            AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=0),
            AnchorSlice(image=np.ones((2, 2), dtype=np.uint8), axis=0, index=1),
        ]

        with self.assertRaisesRegex(ValueError, "latent plane"):
            encode_scale_anchors(
                DownsamplingVAE(),
                anchors,
                volume_size=6,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_anchor_latents_reject_base_anchor_index_outside_base_volume(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=2)

        with self.assertRaisesRegex(ValueError, "index"):
            encode_scale_anchors(
                IdentityVAE(),
                [anchor],
                volume_size=6,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_anchor_latents_reject_full_anchor_index_outside_full_volume(self):
        anchor = AnchorSlice(image=np.zeros((4, 4), dtype=np.uint8), axis=0, index=4)

        with self.assertRaisesRegex(ValueError, "index"):
            encode_scale_anchors(
                IdentityVAE(),
                [anchor],
                volume_size=4,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_anchor_latents_reject_center_that_cannot_align_to_latent_grid(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "align"):
            encode_scale_anchors(
                DownsamplingVAE(),
                [anchor],
                volume_size=4,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_large_anchor_latents_do_not_require_base_center_alignment(self):
        anchor = AnchorSlice(image=np.zeros((4, 4), dtype=np.uint8), axis=0, index=0)

        latent, mask = encode_scale_anchors(
            DownsamplingVAE(),
            [anchor],
            volume_size=4,
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        self.assertEqual(latent.shape, torch.Size([1, 2, 2, 2]))
        self.assertEqual(mask.shape, torch.Size([1, 2, 2, 2]))

    def test_anchor_latents_reject_non_finite_encoded_base_anchor(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "encoded anchor latent.*finite"):
            encode_scale_anchors(
                NonFiniteVAE(),
                [anchor],
                volume_size=4,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_anchor_latents_reject_non_finite_encoded_large_anchor(self):
        anchor = AnchorSlice(image=np.zeros((4, 4), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "encoded anchor latent.*finite"):
            encode_scale_anchors(
                NonFiniteVAE(),
                [anchor],
                volume_size=4,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_shifted_anchor_slices_move_base_index_to_output_index(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=1)

        shifted = shift_anchor_slices([anchor], volume_size=6, base_size=2)

        self.assertEqual(shifted, [(0, 3)])

    def test_scale_anchor_targets_reject_center_that_cannot_align_to_latent_grid(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "align"):
            build_scale_targets(
                DownsamplingVAE(),
                [anchor],
                volume_size=4,
                base_size=2,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
                downsample_factor=2,
            )

    def test_scale_anchor_targets_use_reconstructed_center_patch(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=1)

        targets, masks = build_scale_targets(
            ShiftDecodeVAE(),
            [anchor],
            volume_size=6,
            base_size=2,
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        target = targets[(0, 3)]
        mask = masks[(0, 3)]

        self.assertTrue(torch.allclose(target[2:4, 2:4], torch.full((2, 2), 0.25)))
        self.assertTrue(torch.equal(mask[2:4, 2:4], torch.ones(2, 2)))
        self.assertTrue(torch.equal(mask[:2, :], torch.zeros(2, 6)))

    def test_shifted_anchor_slices_reject_center_that_cannot_align_to_latent_grid(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "align"):
            shift_anchor_slices(
                [anchor],
                volume_size=4,
                base_size=2,
                downsample_factor=2,
            )

    def test_shifted_anchor_slices_reject_index_outside_base_volume(self):
        anchor = AnchorSlice(image=np.zeros((2, 2), dtype=np.uint8), axis=0, index=2)

        with self.assertRaisesRegex(ValueError, "index"):
            shift_anchor_slices([anchor], volume_size=6, base_size=2)


if __name__ == "__main__":
    unittest.main()
