import unittest

import torch

from src.predict.scale import decode_large_latent_volume, refine_large_volume


class DecodeValueVAE(torch.nn.Module):
    image_size = 2
    latent_size = 1
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


class ShiftRefineVAE(torch.nn.Module):
    image_size = 2

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
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 2, self.image_size, self.image_size)


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
        self.assertTrue(torch.allclose(volume, expected.clamp(-1.0, 1.0)))
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
            d_value = 1.0 if z < 2 else 2.0
            for y in range(4):
                h_value = 3.0 if y < 2 else 4.0
                for x in range(4):
                    w_value = 5.0 if x < 2 else 6.0
                    expected[z, y, x] = (d_value + h_value + w_value) / 3.0

        self.assertTrue(torch.allclose(refined, expected.clamp(-1.0, 1.0)))
        self.assertFalse(vae.training)
        self.assertTrue(vae.encode_grad_enabled)
        self.assertTrue(vae.decode_grad_enabled)
        self.assertTrue(all(enabled is False for enabled in vae.encode_grad_enabled))
        self.assertTrue(all(enabled is False for enabled in vae.decode_grad_enabled))

    def test_refine_large_volume_rejects_bad_decode_shape(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            refine_large_volume(
                torch.zeros(2, 2, 2),
                BadRefineVAE(),
                steps=1,
                tile_overlap=0,
            )


if __name__ == "__main__":
    unittest.main()
