import unittest

import torch

from src.models import CustomVAE, DDPM, TimeUNet, TorchFEMMesh
from src.loss import (
    UNetDiffusionLoss,
    compute_diffusivity_loss,
    compute_relative_surface_area,
    compute_sa_loss,
    compute_tpc_loss_ste,
    compute_tpc_torch,
    compute_vf_loss,
    compute_vf_moment_loss,
    compute_volume_fraction,
    diffusion_noise_loss,
    build_grayscale_tpc_target,
    build_grayscale_tpc_targets,
    compute_grayscale_tpc_loss,
    setup_tpc_bins,
)
from src.loss.vae import compute_vae_loss_parts


class LossTest(unittest.TestCase):
    def test_diffusion_noise_loss_is_mse_between_predicted_and_actual_noise(self):
        pred = torch.tensor([0.0, 2.0])
        noise = torch.tensor([1.0, 0.0])

        loss = diffusion_noise_loss(pred, noise)

        self.assertEqual(float(loss), 2.5)

    def test_vae_loss_combines_reconstruction_ssim_and_kl(self):
        recon = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2)
        mu = torch.zeros(1, 4, 1, 1)
        logvar = torch.zeros(1, 4, 1, 1)

        total, parts = compute_vae_loss_parts(recon, target, mu, logvar, kl_weight=0.1, ssim_weight=0.1)

        self.assertEqual(float(parts["reconstruction"]), 1.0)
        self.assertGreater(float(parts["ssim"]), 0.9)
        self.assertEqual(float(parts["kl"]), 0.0)
        self.assertGreater(float(total), float(parts["reconstruction"]))

    def test_vae_kl_matches_reference_input_numel_normalization(self):
        recon = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        mu = torch.ones(1, 4, 1, 1)
        logvar = torch.zeros(1, 4, 1, 1)

        _, parts = compute_vae_loss_parts(recon, target, mu, logvar, kl_weight=1.0, ssim_weight=0.0)

        self.assertEqual(float(parts["kl"]), 0.125)

    def test_compute_tpc_torch_returns_radial_autocorrelation(self):
        mask = torch.ones(4, 4)
        bin_mat, bin_counts = setup_tpc_bins(4, 4, device="cpu")

        tpc = compute_tpc_torch(mask, bin_mat, bin_counts)

        self.assertTrue(torch.allclose(tpc, torch.ones_like(tpc)))

    def test_tpc_loss_ste_matches_reference_target(self):
        masks_p = torch.stack([torch.ones(4, 4), torch.zeros(4, 4)])
        bin_mat, bin_counts = setup_tpc_bins(4, 4, device="cpu")
        target = compute_tpc_torch(torch.ones(4, 4), bin_mat, bin_counts).detach()

        loss = compute_tpc_loss_ste(
            masks_p=masks_p,
            phases=[0],
            tpc_targets={0: target.numpy()},
            bin_mat=bin_mat,
            bin_counts=bin_counts,
            device=torch.device("cpu"),
        )

        self.assertEqual(float(loss), 0.0)

    def test_grayscale_tpc_target_and_loss_match_condition(self):
        condition = torch.rand(1, 1, 8, 8)
        target, bin_mat, bin_counts = build_grayscale_tpc_target(condition)

        loss = compute_grayscale_tpc_loss(condition, target, bin_mat, bin_counts)

        self.assertEqual(target.ndim, 1)
        self.assertEqual(float(loss), 0.0)

    def test_grayscale_tpc_targets_average_multiple_conditions(self):
        first = torch.zeros(1, 1, 8, 8)
        second = torch.ones(1, 1, 8, 8)
        first_target, _, _ = build_grayscale_tpc_target(first)
        second_target, _, _ = build_grayscale_tpc_target(second)

        target, _, _ = build_grayscale_tpc_targets([first, second])

        self.assertTrue(torch.allclose(target, (first_target + second_target) / 2))

    def test_compute_vf_loss_matches_reference_moments(self):
        decoded = torch.full((2, 1, 4, 4), 0.5)

        loss = compute_vf_loss(
            decoded=decoded,
            vf0=0.0,
            vf05=1.0,
            vf1=0.0,
            w_m1=1.0,
            w_m2=1.0,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(compute_volume_fraction(decoded).mean()), 0.5)

    def test_compute_vf_moment_loss_matches_grayscale_moments(self):
        decoded = torch.full((2, 1, 4, 4), 0.5)

        loss = compute_vf_moment_loss(
            decoded=decoded,
            target_mean=0.5,
            target_sqmean=0.25,
        )

        self.assertEqual(float(loss), 0.0)

    def test_surface_area_loss_matches_target(self):
        decoded = torch.zeros(1, 1, 8, 8)
        phases = [0, 1]
        rel_sa = compute_relative_surface_area(decoded, phases)
        targets = {phase: float(rel_sa[i]) for i, phase in enumerate(phases)}

        loss = compute_sa_loss(decoded, targets, phases, device=torch.device("cpu"))

        self.assertEqual(float(loss), 0.0)

    def test_torch_fem_mesh_returns_scalar_diffusivity(self):
        fem = TorchFEMMesh(2, 2, device="cpu")
        mask = torch.ones(2, 2)

        deff = fem(mask)

        self.assertEqual(deff.ndim, 0)
        self.assertGreater(float(deff), 0.0)

    def test_torch_fem_mesh_places_buffers_on_requested_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        fem = TorchFEMMesh(2, 2, device=device)

        self.assertEqual(fem.base_data.device.type, device)
        self.assertEqual(fem.bc_idx.device.type, device)

    def test_diffusivity_loss_matches_targets(self):
        fem = TorchFEMMesh(2, 2, device="cpu")
        masks = torch.zeros(1, 2, 2, 2)
        masks[0, 0] = 1.0
        masks[0, 1] = 0.0
        phases = [0, 1]
        targets = {0: float(fem(masks[0, 0]).detach()), 1: float(fem(masks[0, 1]).detach())}

        loss = compute_diffusivity_loss(
            masks=masks,
            fem_solver=fem,
            rd_targets=targets,
            phases=phases,
            device=torch.device("cpu"),
        )

        self.assertEqual(float(loss), 0.0)

    def test_unet_diffusion_loss_returns_loss_dict(self):
        vae = CustomVAE(latent_ch=4).eval()
        ddpm = DDPM(timesteps=10)
        model = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
        criterion = UNetDiffusionLoss(vae=vae, ddpm=ddpm)
        batch = torch.rand(2, 1, 64, 64)

        loss_dict, loss = criterion(model, batch)

        self.assertEqual(set(loss_dict), {"loss", "diffusion"})
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
