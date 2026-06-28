import unittest

import torch
import torch.nn.functional as F

from src.loss import (
    VAELoss,
    kl_divergence,
    phase_levels,
    phase_logits,
    phase_loss,
    ssim_loss,
    vae_loss,
)


class VAELossTest(unittest.TestCase):
    def test_kl_divergence_is_zero_for_unit_gaussian(self):
        mu = torch.zeros(2, 4, 16, 16)
        logvar = torch.zeros_like(mu)

        kl = kl_divergence(mu, logvar)

        self.assertTrue(torch.allclose(kl, torch.tensor(0.0)))

    def test_ssim_loss_is_zero_for_identical_images(self):
        image = torch.linspace(-1.0, 1.0, 32 * 32).view(1, 1, 32, 32)

        loss = ssim_loss(image, image)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_phase_levels_span_minus_one_to_one(self):
        levels = phase_levels(num_phases=4)

        expected = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
        self.assertTrue(torch.allclose(levels, expected))

    def test_phase_logits_are_larger_for_closer_phase_levels(self):
        recon = torch.tensor([[[[-0.9, 0.8]]]])

        logits = phase_logits(recon, num_phases=3, temperature=0.1)

        self.assertEqual(logits.shape, torch.Size([1, 3, 1, 2]))
        self.assertEqual(int(logits.argmax(dim=1)[0, 0, 0]), 0)
        self.assertEqual(int(logits.argmax(dim=1)[0, 0, 1]), 2)

    def test_phase_loss_matches_distance_based_cross_entropy(self):
        recon = torch.tensor([[[[-0.9, 0.8]]]])
        target = torch.tensor([[[[-1.0, 1.0]]]])
        logits = phase_logits(recon, num_phases=3, temperature=0.1)
        target_index = torch.tensor([[[0, 2]]])

        loss = phase_loss(recon, target, num_phases=3, temperature=0.1)
        expected = F.cross_entropy(logits, target_index)

        self.assertTrue(torch.allclose(loss, expected))

    def test_vae_loss_combines_reconstruction_ssim_phase_and_beta_weighted_kl(self):
        recon = torch.zeros(1, 1, 32, 32)
        target = torch.ones_like(recon)
        mu = torch.ones(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        total, parts = vae_loss(
            recon,
            target,
            mu,
            logvar,
            beta=2.0,
            ssim_weight=0.5,
        )
        expected = (
            parts["reconstruction"]
            + 0.5 * parts["ssim"]
            + 0.1 * parts["phase"]
            + 2.0 * parts["kl"]
        )

        self.assertTrue(torch.allclose(parts["reconstruction"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["kl"], torch.tensor(0.5)))
        self.assertIn("ssim", parts)
        self.assertIn("phase", parts)
        self.assertGreater(parts["ssim"].item(), 0.0)
        self.assertGreater(parts["phase"].item(), 0.0)
        self.assertTrue(torch.allclose(total, expected))

    def test_vae_loss_includes_phase_loss_when_weighted(self):
        recon = torch.tensor([[[[-0.8, 0.2], [0.1, 0.9]]]])
        target = torch.tensor([[[[-1.0, 0.0], [0.0, 1.0]]]])
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        total, parts = vae_loss(
            recon,
            target,
            mu,
            logvar,
            ssim_weight=0.0,
            phase_weight=0.25,
            num_phases=3,
            phase_temperature=0.1,
        )
        expected = (
            parts["reconstruction"]
            + 0.25 * parts["phase"]
            + parts["kl"]
        )

        self.assertIn("phase", parts)
        self.assertGreater(parts["phase"].item(), 0.0)
        self.assertTrue(torch.allclose(total, expected))

    def test_vae_loss_rejects_mismatched_shapes(self):
        recon = torch.zeros(1, 1, 2, 2)
        target = torch.zeros(1, 1, 4, 4)
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        with self.assertRaisesRegex(ValueError, "recon"):
            vae_loss(recon, target, mu, logvar)

        with self.assertRaisesRegex(ValueError, "mu"):
            vae_loss(recon, recon, mu, torch.zeros(1, 4, 2, 2))

        with self.assertRaisesRegex(ValueError, "batch"):
            vae_loss(torch.zeros(2, 1, 2, 2), torch.zeros(2, 1, 2, 2), mu, logvar)

    def test_kl_divergence_rejects_empty_latent(self):
        mu = torch.zeros(0, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        with self.assertRaisesRegex(ValueError, "empty"):
            kl_divergence(mu, logvar)

    def test_vae_loss_rejects_negative_beta(self):
        recon = torch.zeros(1, 1, 2, 2)
        mu = torch.zeros(1, 4, 1, 1)

        with self.assertRaisesRegex(ValueError, "beta"):
            vae_loss(recon, recon, mu, mu, beta=-1.0)

        with self.assertRaisesRegex(ValueError, "ssim_weight"):
            vae_loss(recon, recon, mu, mu, ssim_weight=-1.0)

        with self.assertRaisesRegex(ValueError, "phase_weight"):
            vae_loss(recon, recon, mu, mu, phase_weight=-1.0)

    def test_vae_loss_module_wraps_function(self):
        recon = torch.zeros(1, 1, 32, 32)
        target = torch.ones_like(recon)
        mu = torch.ones(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)
        loss_fn = VAELoss(
            beta=2.0,
            ssim_weight=0.5,
            phase_weight=0.25,
            num_phases=3,
            phase_temperature=0.1,
        )

        total, parts = loss_fn(recon, target, mu, logvar)
        expected = (
            parts["reconstruction"]
            + 0.5 * parts["ssim"]
            + 0.25 * parts["phase"]
            + 2.0 * parts["kl"]
        )

        self.assertTrue(torch.allclose(parts["reconstruction"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(parts["kl"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(total, expected))


if __name__ == "__main__":
    unittest.main()
