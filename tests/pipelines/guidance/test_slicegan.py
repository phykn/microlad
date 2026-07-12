import unittest

import torch

from src.pipelines.guidance.slicegan import (
    build_anchor_references,
    build_diffusion_references,
    multiscale_shape_loss,
    noise_distribution_loss,
    volume_slices,
)


class FakeSampler:
    def sample(self, shape):
        return torch.zeros(shape)


class CategoricalVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 64
        self.latent_size = 2
        self.latent_ch = 1
        self.num_phases = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        probabilities = torch.full(
            (latent.shape[0], 2, 64, 64),
            0.5,
            device=latent.device,
        )
        return probabilities


class SliceGANTest(unittest.TestCase):
    def test_anchor_references_preserve_labels_and_all_dihedral_views(self):
        anchor = torch.zeros(64, 64, dtype=torch.long)
        anchor[:16] = 1

        references = build_anchor_references(anchor, num_phases=2)

        self.assertEqual(references.shape, torch.Size([8, 2, 64, 64]))
        self.assertTrue(torch.equal(references.sum(dim=1), torch.ones(8, 64, 64)))
        self.assertTrue(
            torch.allclose(
                references[:, 1].mean(dim=(1, 2)),
                torch.full((8,), 0.25),
            )
        )

    def test_diffusion_references_are_calibrated_to_target_fraction(self):
        references = build_diffusion_references(
            FakeSampler(),
            CategoricalVAE(),
            target_fraction=torch.tensor([0.75, 0.25]),
            num_phases=2,
            count=2,
        )

        self.assertEqual(references.shape, torch.Size([2, 2, 64, 64]))
        self.assertTrue(
            torch.allclose(
                references.mean(dim=(0, 2, 3)),
                torch.tensor([0.75, 0.25]),
            )
        )

    def test_volume_slices_preserve_every_plane_for_each_axis(self):
        labels = torch.zeros(64, 64, 64, dtype=torch.long)
        labels[1] = 1
        labels[:, 2] = 1
        probabilities = torch.nn.functional.one_hot(
            labels,
            num_classes=2,
        ).movedim(-1, 0).unsqueeze(0).float()

        axis_zero = volume_slices(probabilities, 0, num_phases=2)
        axis_one = volume_slices(probabilities, 1, num_phases=2)
        axis_two = volume_slices(probabilities, 2, num_phases=2)

        self.assertEqual(axis_zero.shape, torch.Size([64, 2, 64, 64]))
        self.assertTrue(torch.equal(axis_zero[1].argmax(dim=0), labels[1]))
        self.assertTrue(torch.equal(axis_one[2].argmax(dim=0), labels[:, 2, :]))
        self.assertTrue(torch.equal(axis_two[3].argmax(dim=0), labels[:, :, 3]))

    def test_shape_and_noise_losses_have_expected_minima(self):
        target = torch.zeros(2, 64, 64)
        target[0, :, :32] = 1.0
        target[1, :, 32:] = 1.0
        changed = target.roll(8, dims=-1)

        self.assertEqual(float(multiscale_shape_loss(target, target)), 0.0)
        self.assertGreater(float(multiscale_shape_loss(changed, target)), 0.0)
        self.assertTrue(torch.isfinite(noise_distribution_loss(torch.randn(1, 32, 4, 4, 4))))


if __name__ == "__main__":
    unittest.main()
