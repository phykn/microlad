import unittest

import torch

from models import CustomVAE, DDPM, SliceConditionedTimeUNet
from training.loss import SliceConditionedDiffusionLoss, diffusion_noise_loss, vae_loss


class LossTest(unittest.TestCase):
    def test_diffusion_noise_loss_is_mse_between_predicted_and_actual_noise(self):
        pred = torch.tensor([0.0, 2.0])
        noise = torch.tensor([1.0, 0.0])

        loss = diffusion_noise_loss(pred, noise)

        self.assertEqual(float(loss), 2.5)

    def test_vae_loss_combines_reconstruction_and_kl(self):
        recon = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2)
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros(1, 4, 1, 1)

        total, parts = vae_loss(recon, target, mu, logvar, kl_weight=0.1)

        self.assertEqual(float(total), 1.0)
        self.assertEqual(float(parts["reconstruction"]), 1.0)
        self.assertEqual(float(parts["kl"]), 0.0)

    def test_slice_conditioned_diffusion_loss_returns_loss_dict(self):
        vae = CustomVAE(latent_ch=4).eval()
        ddpm = DDPM(timesteps=10)
        model = SliceConditionedTimeUNet(latent_ch=4, base_ch=16, time_dim=16, max_slices=64)
        criterion = SliceConditionedDiffusionLoss(vae=vae, ddpm=ddpm)
        batch = {
            "target": torch.rand(2, 1, 64, 64),
            "condition": torch.rand(2, 1, 64, 64),
            "axis": torch.tensor([0, 0]),
            "slice_index": torch.tensor([12, 24]),
        }

        loss_dict, loss = criterion(model, batch)

        self.assertEqual(set(loss_dict), {"loss", "diffusion", "condition_dropout"})
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(float(loss.detach()), 0.0)

    def test_slice_conditioned_diffusion_loss_can_drop_all_conditions(self):
        vae = CustomVAE(latent_ch=4).eval()
        ddpm = DDPM(timesteps=10)
        model = SliceConditionedTimeUNet(latent_ch=4, base_ch=16, time_dim=16, max_slices=64)
        criterion = SliceConditionedDiffusionLoss(vae=vae, ddpm=ddpm, condition_dropout=1.0)
        batch = {
            "target": torch.rand(2, 1, 64, 64),
            "condition": torch.rand(2, 1, 64, 64),
            "axis": torch.tensor([0, 0]),
            "slice_index": torch.tensor([12, 24]),
        }

        loss_dict, loss = criterion(model, batch)

        self.assertEqual(float(loss_dict["condition_dropout"]), 1.0)
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
