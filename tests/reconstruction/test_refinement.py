import unittest

import torch

from src.reconstruction.refinement import three_axis_refinement


class CountingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 2
        self.encode_inputs: list[torch.Tensor] = []
        self.decode_inputs: list[torch.Tensor] = []
        self.grad_enabled: list[bool] = []
        self.decoded_slices = 0

    def encode(self, image: torch.Tensor):
        self.encode_inputs.append(image.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        return image, torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        batch_size = latent.shape[0]
        values = torch.arange(
            self.decoded_slices + 1,
            self.decoded_slices + batch_size + 1,
            dtype=latent.dtype,
            device=latent.device,
        ).view(batch_size, 1, 1, 1)
        self.decoded_slices += batch_size
        return torch.ones_like(latent) * values


class NonFiniteEncodeVAE(CountingVAE):
    def encode(self, image: torch.Tensor):
        self.encode_inputs.append(image.detach().clone())
        self.grad_enabled.append(torch.is_grad_enabled())
        return torch.full_like(image, float("nan")), torch.zeros_like(image)


class NonFiniteDecodeVAE(CountingVAE):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        return torch.full_like(latent, float("nan"))


class PredictRefineTest(unittest.TestCase):
    def test_three_axis_refinement_averages_encode_decode_from_three_axes(self):
        vae = CountingVAE()
        volume = torch.zeros(2, 2, 2)

        refined = three_axis_refinement(volume, vae, steps=1)

        expected = torch.empty(2, 2, 2)
        for z in range(2):
            z_value = 1.0 if z == 0 else 2.0
            for y in range(2):
                y_value = 3.0 if y == 0 else 4.0
                for x in range(2):
                    x_value = 5.0 if x == 0 else 6.0
                    expected[z, y, x] = (z_value + y_value + x_value) / 3.0

        self.assertEqual(refined.shape, torch.Size([2, 2, 2]))
        self.assertTrue(torch.allclose(refined, expected))
        self.assertEqual(len(vae.encode_inputs), 3)
        self.assertEqual(len(vae.decode_inputs), 3)
        self.assertEqual(
            [tuple(image.shape) for image in vae.encode_inputs],
            [(2, 1, 2, 2), (2, 1, 2, 2), (2, 1, 2, 2)],
        )

    def test_three_axis_refinement_runs_without_gradients_and_sets_eval(self):
        vae = CountingVAE()
        vae.train()

        three_axis_refinement(torch.zeros(2, 2, 2), vae, steps=1)

        self.assertFalse(vae.training)
        self.assertEqual(vae.grad_enabled, [False] * 3)

    def test_three_axis_refinement_zero_steps_returns_clamped_input(self):
        vae = CountingVAE()
        volume = torch.tensor([[[-2.0, 0.0], [0.5, 2.0]], [[1.5, -1.5], [0.0, 1.0]]])

        refined = three_axis_refinement(volume, vae, steps=0)

        self.assertTrue(torch.equal(refined, volume.float()))
        self.assertEqual(vae.encode_inputs, [])

    def test_three_axis_refinement_rejects_invalid_volume_shape(self):
        vae = CountingVAE()

        with self.assertRaisesRegex(ValueError, "shape"):
            three_axis_refinement(torch.zeros(1, 2, 2, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "cubic"):
            three_axis_refinement(torch.zeros(2, 3, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "image_size"):
            three_axis_refinement(torch.zeros(3, 3, 3), vae, steps=1)

    def test_three_axis_refinement_rejects_non_integer_steps(self):
        with self.assertRaisesRegex(ValueError, "steps.*integer"):
            three_axis_refinement(torch.zeros(2, 2, 2), CountingVAE(), steps=1.5)

    def test_three_axis_refinement_rejects_non_floating_volume(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            three_axis_refinement(
                torch.zeros(2, 2, 2, dtype=torch.int64),
                CountingVAE(),
                steps=1,
            )

    def test_three_axis_refinement_rejects_non_finite_volume(self):
        with self.assertRaisesRegex(ValueError, "volume.*finite"):
            three_axis_refinement(
                torch.full((2, 2, 2), float("nan")),
                CountingVAE(),
                steps=1,
            )

    def test_three_axis_refinement_rejects_non_finite_encoded_latent(self):
        with self.assertRaisesRegex(ValueError, "encoded.*finite"):
            three_axis_refinement(torch.zeros(2, 2, 2), NonFiniteEncodeVAE(), steps=1)

    def test_three_axis_refinement_rejects_non_finite_decoded_slice(self):
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            three_axis_refinement(torch.zeros(2, 2, 2), NonFiniteDecodeVAE(), steps=1)


if __name__ == "__main__":
    unittest.main()
