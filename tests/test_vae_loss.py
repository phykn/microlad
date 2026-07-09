import unittest

import torch
import torch.nn.functional as F

from src.loss import (
    VAELoss,
    kl_divergence,
    logits_to_phase_values,
    phase_cross_entropy,
    phase_levels,
    phase_logits,
    phase_loss,
    phase_target_indices,
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
        image = torch.linspace(0.0, 2.0, 32 * 32).view(1, 1, 32, 32)

        loss = ssim_loss(image, image, data_range=2.0)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_ssim_loss_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            ssim_loss(torch.empty(0, 1, 32, 32), torch.empty(0, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, "empty"):
            ssim_loss(torch.empty(1, 1, 0, 32), torch.empty(1, 1, 0, 32))

    def test_phase_levels_are_zero_based_phase_indices(self):
        levels = phase_levels(num_phases=4)

        expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(levels, expected))

    def test_phase_logits_are_larger_for_closer_phase_levels(self):
        recon = torch.tensor([[[[0.1, 1.8]]]])

        logits = phase_logits(recon, num_phases=3, temperature=0.1)

        self.assertEqual(logits.shape, torch.Size([1, 3, 1, 2]))
        self.assertEqual(int(logits.argmax(dim=1)[0, 0, 0]), 0)
        self.assertEqual(int(logits.argmax(dim=1)[0, 0, 1]), 2)

    def test_phase_loss_matches_distance_based_cross_entropy(self):
        recon = torch.tensor([[[[0.1, 1.8]]]])
        target = torch.tensor([[[[0.0, 2.0]]]])
        logits = phase_logits(recon, num_phases=3, temperature=0.1)
        target_index = torch.tensor([[[0, 2]]])

        loss = phase_loss(recon, target, num_phases=3, temperature=0.1)
        expected = F.cross_entropy(logits, target_index)

        self.assertTrue(torch.allclose(loss, expected))

    def test_phase_target_indices_maps_phase_values_to_classes(self):
        target = torch.tensor([[[[0.0, 1.0, 2.0]]]])

        indices = phase_target_indices(target, num_phases=3)

        self.assertTrue(torch.equal(indices, torch.tensor([[[0, 1, 2]]])))

    def test_phase_cross_entropy_uses_logit_channels_directly(self):
        logits = torch.tensor(
            [
                [
                    [[4.0, 0.0]],
                    [[0.0, 4.0]],
                    [[0.0, 0.0]],
                ]
            ]
        )
        target = torch.tensor([[[[0.0, 1.0]]]])

        loss = phase_cross_entropy(logits, target, num_phases=3)
        expected = F.cross_entropy(logits, torch.tensor([[[0, 1]]]))

        self.assertTrue(torch.allclose(loss, expected))

    def test_logits_to_phase_values_returns_soft_expected_phase_index_image(self):
        logits = torch.tensor(
            [
                [
                    [[10.0, 0.0]],
                    [[0.0, 10.0]],
                    [[0.0, 0.0]],
                ]
            ]
        )

        values = logits_to_phase_values(logits, num_phases=3)

        self.assertEqual(values.shape, torch.Size([1, 1, 1, 2]))
        self.assertLess(abs(values[0, 0, 0, 0].item()), 0.01)
        self.assertLess(abs(values[0, 0, 0, 1].item() - 1.0), 0.01)

    def test_phase_loss_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            phase_loss(torch.empty(0, 1, 2, 2), torch.empty(0, 1, 2, 2), num_phases=3)
        with self.assertRaisesRegex(ValueError, "empty"):
            phase_loss(torch.empty(1, 1, 0, 2), torch.empty(1, 1, 0, 2), num_phases=3)

    def test_vae_loss_combines_reconstruction_ssim_and_beta_weighted_kl(self):
        logits = torch.zeros(1, 3, 32, 32)
        target = torch.full((1, 1, 32, 32), 2.0)
        mu = torch.ones(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        total, parts = vae_loss(
            logits,
            target,
            mu,
            logvar,
            beta=2.0,
            ssim_weight=0.5,
        )
        expected = (
            parts["reconstruction"]
            + 0.5 * parts["ssim"]
            + 2.0 * parts["kl"]
        )

        self.assertTrue(
            torch.allclose(parts["reconstruction"], torch.tensor(1.0986), atol=1e-4)
        )
        self.assertTrue(torch.allclose(parts["kl"], torch.tensor(0.5)))
        self.assertIn("ssim", parts)
        self.assertNotIn("phase", parts)
        self.assertGreater(parts["ssim"].item(), 0.0)
        self.assertTrue(torch.allclose(total, expected))

    def test_vae_loss_uses_only_reconstruction_ce_without_ssim_or_kl_weights(self):
        logits = torch.zeros(1, 3, 2, 2)
        target = torch.tensor([[[[0.0, 1.0], [1.0, 2.0]]]])
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        total, parts = vae_loss(
            logits,
            target,
            mu,
            logvar,
            ssim_weight=0.0,
            num_phases=3,
        )
        expected = parts["reconstruction"] + parts["kl"]

        self.assertNotIn("phase", parts)
        self.assertTrue(torch.allclose(total, expected))

    def test_vae_loss_rejects_mismatched_shapes(self):
        logits = torch.zeros(1, 3, 2, 2)
        target = torch.zeros(1, 1, 4, 4)
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        with self.assertRaisesRegex(ValueError, "recon"):
            vae_loss(logits, target, mu, logvar)

        with self.assertRaisesRegex(ValueError, "num_phases"):
            vae_loss(torch.zeros(1, 2, 2, 2), torch.zeros(1, 1, 2, 2), mu, logvar)

        with self.assertRaisesRegex(ValueError, "mu"):
            vae_loss(logits, torch.zeros(1, 1, 2, 2), mu, torch.zeros(1, 4, 2, 2))

        with self.assertRaisesRegex(ValueError, "batch"):
            vae_loss(torch.zeros(2, 3, 2, 2), torch.zeros(2, 1, 2, 2), mu, logvar)

    def test_vae_loss_rejects_empty_reconstruction_inputs(self):
        recon = torch.empty(1, 3, 0, 2)
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        with self.assertRaisesRegex(ValueError, "empty"):
            vae_loss(
                recon,
                torch.empty(1, 1, 0, 2),
                mu,
                logvar,
                ssim_weight=0.0,
            )

    def test_kl_divergence_rejects_empty_latent(self):
        mu = torch.zeros(0, 4, 1, 1)
        logvar = torch.zeros_like(mu)

        with self.assertRaisesRegex(ValueError, "empty"):
            kl_divergence(mu, logvar)

    def test_vae_loss_rejects_negative_beta(self):
        recon = torch.zeros(1, 3, 2, 2)
        target = torch.zeros(1, 1, 2, 2)
        mu = torch.zeros(1, 4, 1, 1)

        with self.assertRaisesRegex(ValueError, "beta"):
            vae_loss(recon, target, mu, mu, beta=-1.0)

        with self.assertRaisesRegex(ValueError, "ssim_weight"):
            vae_loss(recon, target, mu, mu, ssim_weight=-1.0)

    def test_vae_loss_module_wraps_function(self):
        recon = torch.zeros(1, 3, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        mu = torch.ones(1, 4, 1, 1)
        logvar = torch.zeros_like(mu)
        loss_fn = VAELoss(
            beta=2.0,
            ssim_weight=0.5,
            num_phases=3,
        )

        total, parts = loss_fn(recon, target, mu, logvar)
        expected = (
            parts["reconstruction"]
            + 0.5 * parts["ssim"]
            + 2.0 * parts["kl"]
        )

        self.assertTrue(
            torch.allclose(parts["reconstruction"], torch.tensor(1.0986), atol=1e-4)
        )
        self.assertTrue(torch.allclose(parts["kl"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(total, expected))


if __name__ == "__main__":
    unittest.main()
