import unittest

import torch
import torch.nn.functional as F

from src.pipelines.reconstruction.refine import refine_probabilities


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


class PhaseOneVAE(CountingVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        probabilities = torch.empty(
            latent.shape[0],
            self.num_phases,
            self.image_size,
            self.image_size,
            dtype=latent.dtype,
            device=latent.device,
        )
        probabilities[:, 0] = 0.05
        probabilities[:, 1] = 0.90
        probabilities[:, 2] = 0.05
        return probabilities


class RefineTest(unittest.TestCase):
    def test_soft_probability_refine_can_change_hard_labels_in_one_step(self):
        probabilities = torch.zeros(1, 3, 2, 2, 2)
        probabilities[:, 0] = 0.51
        probabilities[:, 1] = 0.49

        refined = refine_probabilities(
            probabilities,
            PhaseOneVAE(),
            steps=1,
            strength=0.15,
        )

        self.assertTrue(torch.all(refined.argmax(dim=1) == 1))

    def test_refine_uses_cross_axis_probability_consensus(self):
        probabilities = F.one_hot(
            torch.zeros(2, 2, 2, dtype=torch.long),
            num_classes=3,
        ).movedim(-1, 0).unsqueeze(0).float()

        refined = refine_probabilities(
            probabilities,
            AxisVAE(),
            steps=1,
        ).argmax(dim=1)[0].float()

        self.assertFalse(torch.any(refined == 1.0))
        self.assertTrue(torch.all(refined == 0.0))

    def test_refine_chunks_axis_slices(self):
        vae = BatchedVAE()
        probabilities = F.one_hot(
            torch.zeros(4, 4, 4, dtype=torch.long),
            num_classes=3,
        ).movedim(-1, 0).unsqueeze(0).float()

        refined = refine_probabilities(
            probabilities,
            vae,
            steps=1,
            batch_size=2,
        )

        self.assertEqual(refined.shape, torch.Size([1, 3, 4, 4, 4]))
        self.assertEqual([image.shape[0] for image in vae.encode_inputs], [2] * 6)

    def test_refine_runs_without_gradients_and_sets_eval(self):
        vae = CountingVAE()
        vae.train()
        probabilities = F.one_hot(
            torch.zeros(2, 2, 2, dtype=torch.long),
            num_classes=3,
        ).movedim(-1, 0).unsqueeze(0).float()

        refine_probabilities(probabilities, vae, steps=1)

        self.assertFalse(vae.training)
        self.assertEqual(vae.grad_enabled, [False] * 3)

    def test_refine_zero_steps_normalizes_probabilities(self):
        vae = CountingVAE()
        probabilities = torch.rand(1, 3, 2, 2, 2)

        refined = refine_probabilities(probabilities, vae, steps=0)

        expected = probabilities / probabilities.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(refined, expected))
        self.assertEqual(vae.encode_inputs, [])

    def test_refine_rejects_invalid_options(self):
        probabilities = torch.ones(1, 3, 2, 2, 2)
        vae = CountingVAE()

        for options in ({"steps": 1.5}, {"steps": 1, "batch_size": 1.5}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "integer"):
                    refine_probabilities(probabilities, vae, **options)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            refine_probabilities(probabilities, vae, steps=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            refine_probabilities(probabilities, vae, steps=1, batch_size=0)

    def test_refine_rejects_invalid_probabilities(self):
        vae = CountingVAE()

        with self.assertRaisesRegex(ValueError, "shape"):
            refine_probabilities(torch.zeros(1, 3, 1, 2, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "floating"):
            refine_probabilities(
                torch.zeros(1, 3, 2, 2, 2, dtype=torch.int64),
                vae,
                steps=1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            refine_probabilities(
                torch.full((1, 3, 2, 2, 2), float("nan")),
                vae,
                steps=1,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            refine_probabilities(-torch.ones(1, 3, 2, 2, 2), vae, steps=1)
        with self.assertRaisesRegex(ValueError, "positive phase mass"):
            refine_probabilities(torch.zeros(1, 3, 2, 2, 2), vae, steps=1)

    def test_refine_rejects_invalid_vae_outputs(self):
        probabilities = F.one_hot(
            torch.zeros(2, 2, 2, dtype=torch.long),
            num_classes=3,
        ).movedim(-1, 0).unsqueeze(0).float()

        with self.assertRaisesRegex(ValueError, "encoded.*finite"):
            refine_probabilities(probabilities, NonFiniteEncodeVAE(), steps=1)
        with self.assertRaisesRegex(ValueError, "decoded.*finite"):
            refine_probabilities(probabilities, NonFiniteDecodeVAE(), steps=1)
        with self.assertRaisesRegex(ValueError, "decode_probs output"):
            refine_probabilities(probabilities, BadDecodeVAE(), steps=1)


if __name__ == "__main__":
    unittest.main()
