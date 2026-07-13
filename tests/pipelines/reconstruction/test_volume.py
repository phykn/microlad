import unittest

import torch

from src.pipelines.reconstruction.volume import (
    decode_axis_probs,
    decode_latent,
    decode_latents,
    decode_volume,
    decode_volume_probs,
    generate_initial_volume,
)


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def sample_lmpdd(self, shape, **kwargs):
        shape = tuple(shape)
        self.calls.append(shape)
        return torch.zeros(shape)


class CategoricalVAE(torch.nn.Module):
    latent_ch = 1
    latent_size = 2
    image_size = 4
    downsample_factor = 2
    num_phases = 3

    def __init__(self) -> None:
        super().__init__()
        self.decode_shapes: list[torch.Size] = []
        self.grad_enabled: list[bool] = []

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(latent.shape)
        self.grad_enabled.append(torch.is_grad_enabled())
        probabilities = torch.zeros(
            latent.shape[0],
            self.num_phases,
            self.image_size,
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        )
        probabilities[:, 0] = 1.0
        return probabilities


class AxisVAE(CategoricalVAE):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        phase = 2 if self.calls == 1 else 0
        self.calls += 1
        probabilities = torch.zeros(
            latent.shape[0],
            self.num_phases,
            self.image_size,
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        )
        probabilities[:, phase] = 1.0
        return probabilities


class MissingDecoderVAE(CategoricalVAE):
    decode_probs = None


class BadDecodeVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latent.shape[0],
            self.num_phases - 1,
            self.image_size,
            self.image_size,
        )


class BadSpatialVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latent.shape[0], self.num_phases, 1, self.image_size)


class NonFiniteDecodeVAE(CategoricalVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (
                latent.shape[0],
                self.num_phases,
                self.image_size,
                self.image_size,
            ),
            float("nan"),
            dtype=latent.dtype,
            device=latent.device,
        )


class ZeroDownsampleVAE(CategoricalVAE):
    downsample_factor = 0


class DifferentiableVAE(torch.nn.Module):
    latent_ch = 1
    latent_size = 2
    image_size = 2
    downsample_factor = 1
    num_phases = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.softmax(torch.cat([-latent, latent], dim=1), dim=1)


class PredictVolumeTest(unittest.TestCase):
    def test_decode_latent_returns_values_and_probabilities(self):
        values, probabilities = decode_latent(
            CategoricalVAE(),
            torch.zeros(1, 1, 2, 2),
            num_phases=3,
        )

        self.assertEqual(values.shape, torch.Size([4, 4]))
        self.assertEqual(probabilities.shape, torch.Size([3, 4, 4]))

    def test_decode_latents_returns_batch_values_and_probabilities(self):
        values, probabilities = decode_latents(
            CategoricalVAE(),
            torch.zeros(2, 1, 2, 2),
            num_phases=3,
        )

        self.assertEqual(values.shape, torch.Size([2, 4, 4]))
        self.assertEqual(probabilities.shape, torch.Size([2, 3, 4, 4]))

    def test_decode_latents_requires_matching_categorical_decoder(self):
        latent = torch.zeros(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "match"):
            decode_latents(CategoricalVAE(), latent, num_phases=2)
        with self.assertRaisesRegex(ValueError, "decode_probs"):
            decode_latents(MissingDecoderVAE(), latent, num_phases=3)

    def test_decode_latents_rejects_bad_outputs(self):
        latent = torch.zeros(1, 1, 2, 2)

        for vae in (BadDecodeVAE(), BadSpatialVAE()):
            with self.subTest(vae=type(vae).__name__):
                with self.assertRaisesRegex(ValueError, "shape"):
                    decode_latents(vae, latent, num_phases=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_latents(NonFiniteDecodeVAE(), latent, num_phases=3)

    def test_decode_volume_uses_cross_axis_probability_consensus(self):
        volume = decode_volume(AxisVAE(), torch.zeros(1, 2, 2, 2))

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertFalse(torch.any(volume == 1.0))
        self.assertTrue(torch.all(volume == 0.0))

    def test_decode_volume_probs_returns_normalized_probabilities(self):
        probabilities = decode_volume_probs(
            CategoricalVAE(),
            torch.zeros(1, 2, 2, 2),
        )

        self.assertEqual(probabilities.shape, torch.Size([1, 3, 4, 4, 4]))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1, 4, 4, 4)))

    def test_decode_axis_probs_returns_aligned_axis_probabilities(self):
        probabilities = decode_axis_probs(
            CategoricalVAE(),
            torch.zeros(1, 2, 2, 2),
        )

        self.assertEqual(probabilities.shape, torch.Size([3, 3, 4, 4, 4]))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(3, 4, 4, 4),
            )
        )

    def test_chunked_checkpoint_decode_preserves_values_and_gradients(self):
        latent = torch.randn(1, 2, 2, 2, requires_grad=True)
        expected = decode_volume_probs(DifferentiableVAE(), latent)
        actual = decode_volume_probs(
            DifferentiableVAE(),
            latent,
            plane_batch_size=1,
            checkpoint_gradients=True,
        )

        self.assertTrue(torch.allclose(actual, expected))
        actual[:, 1].mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())

    def test_generate_initial_volume_samples_and_decodes_without_gradients(self):
        sampler = FakeSampler()
        vae = CategoricalVAE()
        vae.train()

        volume = generate_initial_volume(sampler, vae)

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(sampler.calls, [(2, 1, 2, 2)])
        self.assertFalse(vae.training)
        self.assertTrue(all(enabled is False for enabled in vae.grad_enabled))

    def test_decode_volume_rejects_invalid_latent(self):
        vae = CategoricalVAE()

        cases = (
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2, 2),
            torch.zeros(1, 2, 2, 2, dtype=torch.int64),
            torch.full((1, 2, 2, 2), float("inf")),
        )
        for latent in cases:
            with self.subTest(shape=latent.shape, dtype=latent.dtype):
                with self.assertRaises(ValueError):
                    decode_volume(vae, latent)

    def test_decode_volume_rejects_invalid_downsample_factor(self):
        with self.assertRaisesRegex(ValueError, "downsample"):
            decode_volume(ZeroDownsampleVAE(), torch.zeros(1, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
