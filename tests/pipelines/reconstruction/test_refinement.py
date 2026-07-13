import unittest

import torch

from src.pipelines.reconstruction.refinement import refine_volume


class CountingVAE(torch.nn.Module):
    image_size = 2
    num_phases = 3

    def __init__(self) -> None:
        super().__init__()
        self.encode_inputs: list[torch.Tensor] = []
        self.decode_inputs: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []

    def encode(self, image: torch.Tensor):
        self.encode_inputs.append(image.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        return image, torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
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


class NonFiniteEncodeVAE(CountingVAE):
    def encode(self, image: torch.Tensor):
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class NonFiniteDecodeVAE(CountingVAE):
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


class BadDecodeVAE(CountingVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latent.shape[0],
            self.num_phases - 1,
            self.image_size,
            self.image_size,
        )


class AxisVAE(CountingVAE):
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


class BatchedVAE(CountingVAE):
    image_size = 4


class PredictRefineTest(unittest.TestCase):
    def test_refine_volume_uses_cross_axis_probability_consensus(self):
        refined = refine_volume(
            torch.zeros(2, 2, 2),
            AxisVAE(),
            steps=1,
        )

        self.assertFalse(torch.any(refined == 1.0))
        self.assertTrue(torch.all(refined == 0.0))

    def test_refine_volume_chunks_axis_slices(self):
        vae = BatchedVAE()

        refined = refine_volume(
            torch.zeros(4, 4, 4),
            vae,
            steps=1,
            batch_size=2,
        )

        self.assertEqual(refined.shape, torch.Size([4, 4, 4]))
        self.assertEqual([image.shape[0] for image in vae.encode_inputs], [2] * 6)

    def test_refine_volume_runs_without_gradients_and_sets_eval(self):
        vae = CountingVAE()
        vae.train()

        refine_volume(torch.zeros(2, 2, 2), vae, steps=1)

        self.assertFalse(vae.training)
        self.assertEqual(vae.grad_enabled, [False] * 3)

    def test_refine_volume_zero_steps_returns_float_input(self):
        vae = CountingVAE()
        volume = torch.tensor(
            [[[-2.0, 0.0], [0.5, 2.0]], [[1.5, -1.5], [0.0, 1.0]]]
        )

        refined = refine_volume(volume, vae, steps=0)

        self.assertTrue(torch.equal(refined, volume.float()))
        self.assertEqual(vae.encode_inputs, [])

    def test_refine_volume_rejects_invalid_options(self):
        volume = torch.zeros(2, 2, 2)
        vae = CountingVAE()

        for options in ({"steps": 1.5}, {"steps": 1, "batch_size": 1.5}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "integer"):
                    refine_volume(volume, vae, **options)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            refine_volume(volume, vae, steps=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            refine_volume(volume, vae, steps=1, batch_size=0)

    def test_refine_volume_rejects_invalid_volume(self):
        vae = CountingVAE()

        with self.assertRaisesRegex(ValueError, "shape"):
            refine_volume(torch.zeros(1, 2, 2, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "cubic"):
            refine_volume(torch.zeros(2, 3, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "image_size"):
            refine_volume(torch.zeros(3, 3, 3), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "floating"):
            refine_volume(torch.zeros(2, 2, 2, dtype=torch.int64), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            refine_volume(torch.full((2, 2, 2), float("nan")), vae, steps=1)

    def test_refine_volume_rejects_invalid_vae_outputs(self):
        volume = torch.zeros(2, 2, 2)

        with self.assertRaisesRegex(ValueError, "encoded.*finite"):
            refine_volume(volume, NonFiniteEncodeVAE(), steps=1)
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            refine_volume(volume, NonFiniteDecodeVAE(), steps=1)
        with self.assertRaisesRegex(ValueError, "decode_probs output"):
            refine_volume(volume, BadDecodeVAE(), steps=1)


if __name__ == "__main__":
    unittest.main()
