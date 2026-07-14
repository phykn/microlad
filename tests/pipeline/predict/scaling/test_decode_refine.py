import unittest

import torch
import torch.nn.functional as F

from src.pipeline.predict.reconstruction.refine import refine_probabilities
from src.pipeline.predict.reconstruction.volume import decode_volume, decode_volume_probs
from src.pipeline.predict.scaling.decoding import (
    decode_tiled_planes,
    decode_large_volume,
    decode_large_volume_probabilities,
)
from src.pipeline.predict.scaling.refine import refine_large_probabilities
from src.pipeline.predict.scaling.tiles import blend_window, tile_grid


class CategoricalVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1
    num_phases = 2

    def __init__(self) -> None:
        super().__init__()
        self.encode_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []

    def encode(self, image: torch.Tensor):
        self.encode_batch_sizes.append(int(image.shape[0]))
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_batch_sizes.append(int(latent.shape[0]))
        phase_one = torch.sigmoid(latent[:, :1])
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class MeanCategoricalVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_batch_sizes.append(int(latent.shape[0]))
        mean = latent.mean(dim=(1, 2, 3), keepdim=True)
        phase_one = torch.sigmoid(mean).expand(
            latent.shape[0],
            1,
            self.image_size,
            self.image_size,
        )
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class UpsampleCategoricalVAE(CategoricalVAE):
    image_size = 4
    downsample_factor = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_batch_sizes.append(int(latent.shape[0]))
        values = F.interpolate(
            latent[:, :1],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        phase_one = torch.sigmoid(values)
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class BadDecodeVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latent.shape[0], 3, self.image_size, self.image_size)


class NonFiniteDecodeVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latent.shape[0], 2, self.image_size, self.image_size),
            float("nan"),
        )


class NonFiniteEncodeVAE(CategoricalVAE):
    def encode(self, image: torch.Tensor):
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class ScalarVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    downsample_factor = 1
    num_phases = 2

    def encode(self, image: torch.Tensor):
        return image, torch.zeros_like(image)


class ScaleDecodeRefineTest(unittest.TestCase):
    def test_large_decode_matches_base_categorical_consensus(self):
        latent = torch.linspace(-2.0, 2.0, steps=8).view(1, 2, 2, 2)

        base = decode_volume(CategoricalVAE(), latent)
        large = decode_large_volume(
            CategoricalVAE(),
            latent,
            tile_overlap=0,
            batch_size=1,
        )

        self.assertTrue(torch.equal(large, base))

    def test_streamed_probabilities_match_base_interpolation(self):
        latent = torch.linspace(-2.0, 2.0, steps=8).view(1, 2, 2, 2)

        base = decode_volume_probs(UpsampleCategoricalVAE(), latent)
        large = decode_large_volume_probabilities(
            UpsampleCategoricalVAE(),
            latent,
            tile_overlap=0,
            batch_size=1,
        )

        self.assertTrue(torch.allclose(large, base, atol=1e-6, rtol=1e-6))

    def test_large_refinement_matches_base_categorical_consensus(self):
        volume = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )

        probabilities = F.one_hot(
            volume.long(),
            num_classes=2,
        ).movedim(-1, 0).unsqueeze(0).float()
        base_probabilities = refine_probabilities(
            probabilities,
            CategoricalVAE(),
            steps=1,
            batch_size=1,
        )
        large = refine_large_probabilities(
            probabilities,
            CategoricalVAE(),
            tile_overlap=0,
            tile_batch_size=1,
            strength=1.0,
            anchor_strength=1.0,
        )

        self.assertTrue(torch.allclose(large, base_probabilities))

    def test_large_refinement_returns_one_probability_volume(self):
        volume = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )

        probabilities = F.one_hot(
            volume.long(),
            num_classes=2,
        ).movedim(-1, 0).unsqueeze(0).float()
        refined = refine_large_probabilities(
            probabilities,
            CategoricalVAE(),
            tile_overlap=0,
            tile_batch_size=1,
        )

        self.assertEqual(refined.shape, probabilities.shape)
        self.assertTrue(torch.isfinite(refined).all())

    def test_large_decode_limits_plane_batch_size(self):
        vae = CategoricalVAE()
        probabilities = decode_large_volume_probabilities(
            vae,
            torch.zeros(1, 4, 4, 4),
            tile_overlap=0,
            batch_size=2,
        )

        self.assertEqual(probabilities.shape, torch.Size([1, 2, 4, 4, 4]))
        self.assertLessEqual(max(vae.decode_batch_sizes), 2)
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(1, 4, 4, 4))
        )

    def test_full_and_batched_decode_have_identical_meaning(self):
        latent = torch.linspace(-1.0, 1.0, steps=64).view(1, 4, 4, 4)
        batched = decode_large_volume_probabilities(
            CategoricalVAE(),
            latent,
            tile_overlap=1,
            batch_size=1,
        )
        full = decode_large_volume_probabilities(
            CategoricalVAE(),
            latent,
            tile_overlap=1,
            batch_size=None,
        )

        self.assertTrue(torch.allclose(full, batched, atol=1e-6, rtol=1e-6))

    def test_tiled_probability_decode_uses_weighted_blending(self):
        vae = MeanCategoricalVAE()
        latent = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)

        decoded = decode_tiled_planes(
            vae,
            latent,
            tile_overlap=1,
            num_phases=2,
        )

        expected = torch.zeros(4, 4)
        weights = torch.zeros_like(expected)
        window = blend_window(2, 2, device=latent.device, dtype=latent.dtype)
        for row, col in tile_grid(4, 4, tile_size=2, overlap=1):
            tile = latent[:, :, row : row + 2, col : col + 2]
            probability = torch.sigmoid(tile.mean())
            expected[row : row + 2, col : col + 2] += probability * window
            weights[row : row + 2, col : col + 2] += window

        self.assertTrue(torch.allclose(decoded[0, 1], expected / weights))

    def test_tiled_probability_decode_limits_vae_batch_size(self):
        vae = CategoricalVAE()
        latent = torch.zeros(5, 1, 4, 4)

        batched = decode_tiled_planes(
            vae,
            latent,
            tile_overlap=1,
            num_phases=2,
            batch_size=2,
        )
        full = decode_tiled_planes(
            CategoricalVAE(),
            latent,
            tile_overlap=1,
            num_phases=2,
        )

        self.assertLessEqual(max(vae.decode_batch_sizes), 2)
        self.assertTrue(torch.equal(batched, full))

    def test_large_refinement_batches_tiles_without_changing_result(self):
        volume = torch.linspace(-0.5, 0.5, steps=64).view(4, 4, 4)
        probabilities = F.one_hot(
            (volume > 0).long(),
            num_classes=2,
        ).movedim(-1, 0).unsqueeze(0).float()
        single = refine_large_probabilities(
            probabilities,
            CategoricalVAE(),
            tile_overlap=1,
            tile_batch_size=1,
        )
        batched_vae = CategoricalVAE()
        batched = refine_large_probabilities(
            probabilities,
            batched_vae,
            tile_overlap=1,
            tile_batch_size=3,
        )

        self.assertTrue(torch.allclose(single, batched))
        self.assertLessEqual(max(batched_vae.encode_batch_sizes), 3)
        self.assertEqual(
            batched_vae.encode_batch_sizes,
            batched_vae.decode_batch_sizes,
        )

    def test_decode_rejects_invalid_inputs(self):
        cases = (
            (CategoricalVAE(), torch.zeros(1, 2, 2, 2, dtype=torch.long), "floating"),
            (CategoricalVAE(), torch.full((1, 2, 2, 2), float("nan")), "finite"),
            (ScalarVAE(), torch.zeros(1, 2, 2, 2), "decode_probs"),
            (BadDecodeVAE(), torch.zeros(1, 2, 2, 2), "decode_probs"),
            (NonFiniteDecodeVAE(), torch.zeros(1, 2, 2, 2), "finite"),
        )
        for vae, latent, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    decode_large_volume(
                        vae,
                        latent,
                        tile_overlap=0,
                        batch_size=1,
                    )

        with self.assertRaisesRegex(ValueError, "batch_size"):
            decode_large_volume(
                CategoricalVAE(),
                torch.zeros(1, 2, 2, 2),
                tile_overlap=0,
                batch_size=0,
            )

    def test_refinement_rejects_invalid_inputs(self):
        cases = (
            (torch.zeros(1, 2, 2, 2, 2, dtype=torch.long), CategoricalVAE(), "floating"),
            (torch.full((1, 2, 2, 2, 2), float("nan")), CategoricalVAE(), "finite"),
            (torch.full((1, 2, 2, 2, 2), 0.5), ScalarVAE(), "decode_probs"),
            (torch.full((1, 2, 2, 2, 2), 0.5), NonFiniteEncodeVAE(), "encoded latent"),
            (torch.full((1, 2, 2, 2, 2), 0.5), NonFiniteDecodeVAE(), "decoded probabilities"),
        )
        for probabilities, vae, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    refine_large_probabilities(
                        probabilities,
                        vae,
                        tile_overlap=0,
                        tile_batch_size=1,
                    )

        with self.assertRaisesRegex(ValueError, "tile_batch_size"):
            refine_large_probabilities(
                torch.full((1, 2, 2, 2, 2), 0.5),
                CategoricalVAE(),
                tile_batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
