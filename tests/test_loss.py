import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.inference import DiffusivitySolver
from src.models import DDPM, PatchVAE, TimeUNet
from src.loss import (
    UNetDiffusionLoss,
    VAELoss,
    compute_diffusivity_loss,
    compute_relative_surface_area,
    compute_surface_area_loss,
    compute_tpc,
    compute_gray_mean,
    compute_gray_moment_loss,
    soft_gray_level_masks,
    build_grayscale_tpc_target,
    build_grayscale_tpc_targets,
    compute_grayscale_tpc_loss,
    build_tpc_bins,
)
from src.loss.objective import compute_vae_loss_parts


class LossTest(unittest.TestCase):
    def test_pytorch_msssim_is_required_for_vae_loss(self):
        code = """
import importlib.abc
import sys

class BlockPytorchMSSSIM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pytorch_msssim" or fullname.startswith("pytorch_msssim."):
            raise ImportError("blocked pytorch_msssim")
        return None

sys.meta_path.insert(0, BlockPytorchMSSSIM())

try:
    import src.loss.objective
except ImportError as error:
    raise SystemExit(0 if "pytorch_msssim" in str(error) else 2)
raise SystemExit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )

        self.assertEqual(result.returncode, 0)

    def test_vae_loss_combines_reconstruction_ssim_and_kl(self):
        recon = torch.zeros(1, 1, 16, 16)
        target = torch.ones(1, 1, 16, 16)
        mu = torch.zeros(1, 4, 4, 4)
        logvar = torch.zeros(1, 4, 4, 4)

        total, parts = compute_vae_loss_parts(
            recon, target, mu, logvar, kl_weight=0.1, ssim_weight=0.1
        )

        self.assertEqual(float(parts["reconstruction"]), 1.0)
        self.assertGreater(float(parts["ssim"]), 0.9)
        self.assertEqual(float(parts["kl"]), 0.0)
        self.assertGreater(float(total), float(parts["reconstruction"]))

    def test_vae_kl_matches_reference_input_numel_normalization(self):
        recon = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        mu = torch.ones(1, 4, 4, 4)
        logvar = torch.zeros(1, 4, 4, 4)

        _, parts = compute_vae_loss_parts(
            recon, target, mu, logvar, kl_weight=1.0, ssim_weight=0.0
        )

        self.assertEqual(float(parts["kl"]), 0.125)

    def test_vae_loss_parts_keep_grad_until_criterion_formats_metrics(self):
        recon = torch.zeros(1, 1, 16, 16, requires_grad=True)
        target = torch.ones(1, 1, 16, 16)
        mu = torch.zeros(1, 4, 4, 4, requires_grad=True)
        logvar = torch.zeros(1, 4, 4, 4, requires_grad=True)

        total, parts = compute_vae_loss_parts(recon, target, mu, logvar)

        self.assertTrue(total.requires_grad)
        self.assertTrue(parts["reconstruction"].requires_grad)
        self.assertTrue(parts["ssim"].requires_grad)
        self.assertTrue(parts["kl"].requires_grad)

    def test_vae_criterion_returns_detached_metrics_and_train_loss(self):
        criterion = VAELoss()
        model = PatchVAE(latent_ch=4)
        batch = torch.rand(2, 1, 64, 64)

        loss, metrics = criterion(model, batch)

        self.assertEqual(set(metrics), {"reconstruction", "ssim", "kl"})
        self.assertTrue(loss.requires_grad)
        self.assertFalse(metrics["reconstruction"].requires_grad)
        self.assertFalse(metrics["ssim"].requires_grad)
        self.assertFalse(metrics["kl"].requires_grad)

    def test_compute_tpc_returns_radial_autocorrelation(self):
        mask = torch.ones(4, 4)
        bin_mat, bin_counts = build_tpc_bins(4, 4, device="cpu")

        tpc = compute_tpc(mask, bin_mat, bin_counts)

        self.assertTrue(torch.allclose(tpc, torch.ones_like(tpc)))

    def test_tpc_bins_reject_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "height"):
            build_tpc_bins(0, 4, device="cpu")
        with self.assertRaisesRegex(ValueError, "width"):
            build_tpc_bins(4, 0, device="cpu")

    def test_grayscale_tpc_target_and_loss_match_condition(self):
        condition = torch.rand(1, 1, 8, 8)
        target, bin_mat, bin_counts = build_grayscale_tpc_target(condition)

        loss = compute_grayscale_tpc_loss(condition, target, bin_mat, bin_counts)

        self.assertEqual(target.ndim, 1)
        self.assertEqual(float(loss), 0.0)

    def test_grayscale_tpc_loss_rejects_target_length_mismatch(self):
        condition = torch.rand(1, 1, 8, 8)
        target, bin_mat, bin_counts = build_grayscale_tpc_target(condition)

        with self.assertRaisesRegex(ValueError, "target length"):
            compute_grayscale_tpc_loss(condition, target[:-1], bin_mat, bin_counts)

    def test_grayscale_tpc_targets_average_multiple_conditions(self):
        first = torch.zeros(1, 1, 8, 8)
        second = torch.ones(1, 1, 8, 8)
        first_target, _, _ = build_grayscale_tpc_target(first)
        second_target, _, _ = build_grayscale_tpc_target(second)

        target, _, _ = build_grayscale_tpc_targets([first, second])

        self.assertTrue(torch.allclose(target, (first_target + second_target) / 2))

    def test_compute_gray_mean_returns_average_intensity(self):
        decoded = torch.full((2, 1, 4, 4), 0.5)

        mean = compute_gray_mean(decoded)

        self.assertTrue(torch.allclose(mean, torch.full((2,), 0.5)))

    def test_gray_descriptor_losses_require_gray_image_batches(self):
        with self.assertRaisesRegex(ValueError, "gray image"):
            compute_gray_mean(torch.zeros(1, 2, 4, 4))
        with self.assertRaisesRegex(ValueError, "gray image"):
            soft_gray_level_masks(torch.zeros(4, 4), [0, 1])

    def test_soft_gray_level_masks_rejects_invalid_levels_and_beta(self):
        image = torch.zeros(1, 1, 4, 4)

        with self.assertRaisesRegex(ValueError, "levels"):
            soft_gray_level_masks(image, [])
        with self.assertRaisesRegex(ValueError, "beta"):
            soft_gray_level_masks(image, [0, 1], beta=0.0)

    def test_compute_gray_moment_loss_matches_grayscale_moments(self):
        decoded = torch.full((2, 1, 4, 4), 0.5)

        loss = compute_gray_moment_loss(
            decoded=decoded,
            target_mean=0.5,
            target_sqmean=0.25,
        )

        self.assertEqual(float(loss), 0.0)

    def test_surface_area_loss_matches_target(self):
        decoded = torch.zeros(1, 1, 8, 8)
        gray_levels = [0, 1]
        rel_sa = compute_relative_surface_area(decoded, gray_levels)
        targets = {level: float(rel_sa[i]) for i, level in enumerate(gray_levels)}

        loss = compute_surface_area_loss(
            decoded, targets, gray_levels, device=torch.device("cpu")
        )

        self.assertEqual(float(loss), 0.0)

    def test_diffusivity_solver_returns_scalar_diffusivity(self):
        solver = DiffusivitySolver(2, 2, device="cpu")
        mask = torch.ones(2, 2)

        deff = solver(mask)

        self.assertEqual(deff.ndim, 0)
        self.assertGreater(float(deff), 0.0)

    def test_diffusivity_solver_rejects_invalid_setup_values(self):
        with self.assertRaisesRegex(ValueError, "height"):
            DiffusivitySolver(0, 2, device="cpu")
        with self.assertRaisesRegex(ValueError, "width"):
            DiffusivitySolver(2, 0, device="cpu")
        with self.assertRaisesRegex(ValueError, "low_cond"):
            DiffusivitySolver(2, 2, low_cond=-0.1, device="cpu")
        with self.assertRaisesRegex(ValueError, "low_cond"):
            DiffusivitySolver(2, 2, low_cond=1.1, device="cpu")

    def test_diffusivity_solver_places_buffers_on_requested_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        solver = DiffusivitySolver(2, 2, device=device)

        self.assertEqual(solver.base_data.device.type, device)
        self.assertEqual(solver.bc_idx.device.type, device)

    def test_diffusivity_loss_matches_targets(self):
        solver = DiffusivitySolver(2, 2, device="cpu")
        masks = torch.zeros(1, 2, 2, 2)
        masks[0, 0] = 1.0
        masks[0, 1] = 0.0
        gray_levels = [0, 1]
        targets = {
            0: float(solver(masks[0, 0]).detach()),
            1: float(solver(masks[0, 1]).detach()),
        }

        loss = compute_diffusivity_loss(
            masks=masks,
            diffusivity_solver=solver,
            diffusivity_targets=targets,
            levels=gray_levels,
            device=torch.device("cpu"),
        )

        self.assertEqual(float(loss), 0.0)

    def test_diffusivity_loss_rejects_invalid_levels(self):
        solver = DiffusivitySolver(2, 2, device="cpu")
        masks = torch.zeros(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "levels"):
            compute_diffusivity_loss(
                masks=masks,
                diffusivity_solver=solver,
                diffusivity_targets={},
                levels=[],
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "mask channels"):
            compute_diffusivity_loss(
                masks=masks,
                diffusivity_solver=solver,
                diffusivity_targets={0: 0.0, 1: 0.0},
                levels=[0, 1],
                device=torch.device("cpu"),
            )

    def test_diffusivity_loss_resizes_masks_to_solver_shape(self):
        solver = DiffusivitySolver(2, 2, device="cpu")
        masks = torch.zeros(1, 2, 4, 4)
        masks[0, 0, :, :2] = 1.0
        masks[0, 1, :, 2:] = 1.0
        resized = F.interpolate(
            masks, size=(2, 2), mode="bilinear", align_corners=False
        )
        gray_levels = [0, 1]
        targets = {
            0: float(solver(resized[0, 0]).detach()),
            1: float(solver(resized[0, 1]).detach()),
        }

        loss = compute_diffusivity_loss(
            masks=masks,
            diffusivity_solver=solver,
            diffusivity_targets=targets,
            levels=gray_levels,
            device=torch.device("cpu"),
        )

        self.assertEqual(float(loss), 0.0)

    def test_unet_diffusion_loss_returns_train_loss_without_duplicate_metric(self):
        vae = PatchVAE(latent_ch=4).eval()
        ddpm = DDPM(timesteps=10)
        model = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
        criterion = UNetDiffusionLoss(vae=vae, ddpm=ddpm)
        batch = torch.rand(2, 1, 64, 64)

        loss, metrics = criterion(model, batch)

        self.assertEqual(metrics, {})
        self.assertEqual(loss.ndim, 0)
        self.assertGreater(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
