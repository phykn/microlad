import unittest

import torch

from src.predict.volume import decode_latent_volume, generate_initial_volume


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def sample_lmpdd(self, shape, **kwargs):
        shape = tuple(shape)
        self.calls.append(shape)
        return torch.zeros(shape)


class CountingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_ch = 1
        self.latent_size = 2
        self.image_size = 4
        self.downsample_factor = 2
        self.decode_shapes: list[torch.Size] = []
        self.grad_enabled: list[bool] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(latent.shape)
        self.grad_enabled.append(torch.is_grad_enabled())
        value = float(len(self.decode_shapes))
        return torch.full(
            (latent.shape[0], 1, self.image_size, self.image_size),
            value,
            dtype=torch.float32,
            device=latent.device,
        )


class BadVAE(CountingVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latent.shape[0], 2, self.image_size, self.image_size)


class BadSpatialVAE(CountingVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latent.shape[0], 1, 1, self.image_size)


class PredictVolumeTest(unittest.TestCase):
    def test_decode_latent_volume_averages_three_axis_decodes(self):
        vae = CountingVAE()
        latent = torch.zeros(1, 2, 2, 2)

        volume = decode_latent_volume(vae, latent)

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
        self.assertGreater(len(vae.decode_shapes), 0)
        self.assertTrue(
            all(shape[1:] == torch.Size([1, 2, 2]) for shape in vae.decode_shapes)
        )
        self.assertTrue(all(enabled is False for enabled in vae.grad_enabled))

    def test_generate_initial_volume_uses_lmpdd_sampler_and_multi_axis_decode(self):
        sampler = FakeSampler()
        vae = CountingVAE()
        vae.train()

        volume = generate_initial_volume(sampler, vae)

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertFalse(vae.training)

    def test_generate_initial_volume_rejects_size_that_does_not_match_vae(self):
        with self.assertRaisesRegex(ValueError, "size"):
            generate_initial_volume(FakeSampler(), CountingVAE(), size=8)

    def test_decode_latent_volume_rejects_bad_latent_shape(self):
        vae = CountingVAE()

        with self.assertRaisesRegex(ValueError, "latent"):
            decode_latent_volume(vae, torch.zeros(2, 2, 2))
        with self.assertRaisesRegex(ValueError, "latent"):
            decode_latent_volume(vae, torch.zeros(2, 2, 2, 2))

    def test_decode_latent_volume_rejects_bad_decode_shape(self):
        with self.assertRaisesRegex(ValueError, "decode"):
            decode_latent_volume(BadVAE(), torch.zeros(1, 2, 2, 2))

    def test_decode_latent_volume_rejects_bad_decode_spatial_shape(self):
        with self.assertRaisesRegex(ValueError, "spatial shape"):
            decode_latent_volume(BadSpatialVAE(), torch.zeros(1, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
