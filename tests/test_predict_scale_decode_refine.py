import unittest

import torch

from src.scaling.blending import blend_window
from src.scaling.decoding import decode_large_latent_volume
from src.scaling.refinement import refine_large_volume
from src.scaling.decoding import _decode_tiled_plane
from src.scaling.refinement import _refine_tiled_plane
from src.scaling.tiles import tile_grid


class DecodeValueVAE(torch.nn.Module):
    image_size = 4
    latent_size = 2
    latent_ch = 1
    downsample_factor = 2

    def __init__(self) -> None:
        super().__init__()
        self.decode_grad_enabled: list[bool] = []
        self.decode_calls = 0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_grad_enabled.append(torch.is_grad_enabled())
        self.decode_calls += 1
        return torch.full(
            (latent.shape[0], 1, self.image_size, self.image_size),
            float(self.decode_calls),
            dtype=latent.dtype,
            device=latent.device,
        )


class BadDecodeVAE(DecodeValueVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2, self.image_size, self.image_size)


class NonFiniteDecodeVAE(DecodeValueVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latent.shape[0], 1, self.image_size, self.image_size),
            float("nan"),
            dtype=latent.dtype,
            device=latent.device,
        )


class MeanDecodeVAE(DecodeValueVAE):
    image_size = 3
    latent_size = 3
    downsample_factor = 1

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        mean = latent.mean(dim=(1, 2, 3), keepdim=True)
        return mean.expand(latent.shape[0], 1, self.image_size, self.image_size)


class InconsistentScaleVAE(DecodeValueVAE):
    image_size = 3
    latent_size = 2
    downsample_factor = 2

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (latent.shape[0], 1, self.image_size, self.image_size),
            dtype=latent.dtype,
            device=latent.device,
        )


class ShiftRefineVAE(torch.nn.Module):
    image_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.encode_grad_enabled: list[bool] = []
        self.decode_grad_enabled: list[bool] = []
        self.decode_calls = 0

    def encode(self, image: torch.Tensor):
        self.encode_grad_enabled.append(torch.is_grad_enabled())
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_grad_enabled.append(torch.is_grad_enabled())
        self.decode_calls += 1
        return torch.full_like(latent, float(self.decode_calls))


class BadRefineVAE(ShiftRefineVAE):
    image_size = 2

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2, self.image_size, self.image_size)


class NonFiniteEncodeRefineVAE(ShiftRefineVAE):
    image_size = 2

    def encode(self, image: torch.Tensor):
        self.encode_grad_enabled.append(torch.is_grad_enabled())
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class NonFiniteDecodeRefineVAE(ShiftRefineVAE):
    image_size = 2

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_grad_enabled.append(torch.is_grad_enabled())
        return torch.full_like(latent, float("nan"))


class AffineRefineVAE(torch.nn.Module):
    image_size = 3

    def __init__(self) -> None:
        super().__init__()
        self.encode_batch_sizes: list[int] = []
        self.decode_batch_sizes: list[int] = []

    def encode(self, image: torch.Tensor):
        self.encode_batch_sizes.append(int(image.shape[0]))
        return image + 0.25, torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_batch_sizes.append(int(latent.shape[0]))
        return latent * 0.5


class LocalPatternRefineVAE(torch.nn.Module):
    image_size = 3

    def encode(self, image: torch.Tensor):
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


class PredictScaleDecodeRefineTest(unittest.TestCase):
    def test_decode_large_latent_volume_averages_three_axes_without_gradients(self):
        vae = DecodeValueVAE()
        vae.train()
        latent = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2)

        volume = decode_large_latent_volume(vae, latent, tile_overlap=0)

        expected = torch.empty(4, 4, 4)
        for z in range(4):
            d_value = 1.0 if z < 2 else 2.0
            for y in range(4):
                h_value = 3.0 if y < 2 else 4.0
                for x in range(4):
                    w_value = 5.0 if x < 2 else 6.0
                    expected[z, y, x] = (d_value + h_value + w_value) / 3.0

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertTrue(torch.allclose(volume, expected))
        self.assertFalse(vae.training)
        self.assertTrue(vae.decode_grad_enabled)
        self.assertTrue(all(enabled is False for enabled in vae.decode_grad_enabled))

    def test_decode_large_latent_volume_rejects_bad_decode_shape(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            decode_large_latent_volume(
                BadDecodeVAE(),
                torch.zeros(1, 2, 2, 2),
                tile_overlap=0,
            )

    def test_decode_large_latent_volume_rejects_non_floating_latent(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            decode_large_latent_volume(
                DecodeValueVAE(),
                torch.zeros(1, 2, 2, 2, dtype=torch.int64),
                tile_overlap=0,
            )

    def test_decode_large_latent_volume_rejects_non_finite_latent(self):
        with self.assertRaisesRegex(ValueError, "latent volume.*finite"):
            decode_large_latent_volume(
                DecodeValueVAE(),
                torch.full((1, 2, 2, 2), float("nan")),
                tile_overlap=0,
            )

    def test_decode_large_latent_volume_rejects_non_finite_decode_output(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            decode_large_latent_volume(
                NonFiniteDecodeVAE(),
                torch.zeros(1, 2, 2, 2),
                tile_overlap=0,
            )

    def test_decode_large_latent_volume_rejects_inconsistent_vae_scale(self):
        with self.assertRaisesRegex(ValueError, "image_size"):
            decode_large_latent_volume(
                InconsistentScaleVAE(),
                torch.zeros(1, 2, 2, 2),
                tile_overlap=0,
            )

    def test_decode_tiled_plane_uses_image_space_weighted_blending(self):
        vae = MeanDecodeVAE()
        latent_plane = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

        decoded = _decode_tiled_plane(vae, latent_plane, tile_overlap=2)

        expected = torch.zeros(4, 4)
        weight_sum = torch.zeros_like(expected)
        window = blend_window(
            vae.image_size,
            vae.image_size,
            device=latent_plane.device,
            dtype=latent_plane.dtype,
        )

        for row, col in tile_grid(4, 4, tile_size=3, overlap=2):
            tile = latent_plane[:, row : row + 3, col : col + 3]
            decoded_tile = torch.full((3, 3), float(tile.mean()))
            expected[row : row + 3, col : col + 3] += decoded_tile * window
            weight_sum[row : row + 3, col : col + 3] += window

        self.assertTrue(torch.allclose(decoded, expected / weight_sum))

    def test_refine_large_volume_runs_three_axis_refinement_without_gradients(self):
        vae = ShiftRefineVAE()
        vae.train()

        refined = refine_large_volume(
            torch.zeros(4, 4, 4),
            vae,
            steps=1,
            tile_overlap=0,
        )

        expected = torch.empty(4, 4, 4)
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    d_value = float(z + 1)
                    h_value = float(y + 5)
                    w_value = float(x + 9)
                    expected[z, y, x] = (d_value + h_value + w_value) / 3.0

        self.assertTrue(torch.allclose(refined, expected))
        self.assertFalse(vae.training)
        self.assertTrue(vae.encode_grad_enabled)
        self.assertTrue(vae.decode_grad_enabled)
        self.assertTrue(all(enabled is False for enabled in vae.encode_grad_enabled))
        self.assertTrue(all(enabled is False for enabled in vae.decode_grad_enabled))

    def test_refine_large_volume_batches_tiles_without_changing_chunk_result(self):
        volume = torch.linspace(-0.5, 0.5, steps=64).view(4, 4, 4)

        single = refine_large_volume(
            volume,
            AffineRefineVAE(),
            steps=1,
            tile_overlap=2,
            tile_batch_size=1,
        )
        batched_vae = AffineRefineVAE()
        batched = refine_large_volume(
            volume,
            batched_vae,
            steps=1,
            tile_overlap=2,
            tile_batch_size=3,
        )

        expected = volume * 0.5 + 0.125

        self.assertTrue(torch.allclose(single, batched))
        self.assertTrue(torch.allclose(batched, expected))
        self.assertIn(3, batched_vae.encode_batch_sizes)
        self.assertLessEqual(max(batched_vae.encode_batch_sizes), 3)
        self.assertEqual(batched_vae.encode_batch_sizes, batched_vae.decode_batch_sizes)

    def test_refine_large_volume_rejects_invalid_tile_batch_size(self):
        with self.assertRaisesRegex(ValueError, "tile_batch_size"):
            refine_large_volume(
                torch.zeros(4, 4, 4),
                AffineRefineVAE(),
                steps=1,
                tile_overlap=2,
                tile_batch_size=0,
            )

    def test_refine_tiled_plane_uses_weighted_blending(self):
        vae = LocalPatternRefineVAE()

        refined = _refine_tiled_plane(
            torch.zeros(4, 4),
            vae,
            tile_overlap=2,
            tile_batch_size=2,
        )

        expected = torch.zeros(4, 4)
        weight_sum = torch.zeros_like(expected)
        window = blend_window(
            vae.image_size,
            vae.image_size,
            device=refined.device,
            dtype=refined.dtype,
        )
        pattern = vae.decode(torch.zeros(1, 1, 3, 3))[0, 0]

        for row, col in tile_grid(4, 4, tile_size=3, overlap=2):
            expected[row : row + 3, col : col + 3] += pattern * window
            weight_sum[row : row + 3, col : col + 3] += window

        self.assertTrue(torch.allclose(refined, expected / weight_sum))

    def test_refine_large_volume_rejects_bad_decode_shape(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            refine_large_volume(
                torch.zeros(2, 2, 2),
                BadRefineVAE(),
                steps=1,
                tile_overlap=0,
            )

    def test_refine_large_volume_rejects_non_floating_volume(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            refine_large_volume(
                torch.zeros(2, 2, 2, dtype=torch.int64),
                ShiftRefineVAE(),
                steps=1,
                tile_overlap=0,
            )

    def test_refine_large_volume_rejects_non_finite_volume(self):
        with self.assertRaisesRegex(ValueError, "volume.*finite"):
            refine_large_volume(
                torch.full((2, 2, 2), float("nan")),
                ShiftRefineVAE(),
                steps=1,
                tile_overlap=0,
            )

    def test_refine_large_volume_rejects_empty_volume(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            refine_large_volume(
                torch.empty(0, 0, 0),
                ShiftRefineVAE(),
                steps=1,
                tile_overlap=0,
            )

    def test_refine_large_volume_rejects_non_finite_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "encoded.*finite"):
            refine_large_volume(
                torch.zeros(2, 2, 2),
                NonFiniteEncodeRefineVAE(),
                steps=1,
                tile_overlap=0,
            )

    def test_refine_large_volume_rejects_non_finite_decoded_tile(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            refine_large_volume(
                torch.zeros(2, 2, 2),
                NonFiniteDecodeRefineVAE(),
                steps=1,
                tile_overlap=0,
            )


if __name__ == "__main__":
    unittest.main()
