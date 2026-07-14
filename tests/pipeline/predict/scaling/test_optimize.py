import unittest

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.modeling.diffusion import DDPMProcess
from src.pipeline.predict.scaling.decoding import (
    decode_anchor_patch,
    decode_large_volume_probabilities,
)
from src.pipeline.predict.scaling.optimize import optimize_large_latent


class TinyVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        phase_one = torch.sigmoid(latent)
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class ContextVAE(TinyVAE):
    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        mean = latent.mean(dim=(1, 2, 3), keepdim=True)
        phase_one = torch.sigmoid(mean).expand(
            latent.shape[0],
            1,
            self.image_size,
            self.image_size,
        )
        return torch.cat([1.0 - phase_one, phase_one], dim=1)


class UpsampleContextVAE(ContextVAE):
    image_size = 4
    downsample_factor = 2


class ThreePhaseVAE(TinyVAE):
    num_phases = 3

    def encode(self, image: torch.Tensor):
        raise AssertionError("scale guidance must not optimize scalar phase images")

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat([-latent, torch.zeros_like(latent), latent], dim=1).softmax(
            dim=1
        )


class ZeroNoise(torch.nn.Module):
    def forward(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(values)


class RecordingNoise(ZeroNoise):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []

    def forward(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(values.shape[0]))
        return super().forward(values, timesteps)


class RecordingCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []
        self.channel_sizes = []

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(probabilities.shape[0]))
        self.channel_sizes.append(int(probabilities.shape[1]))
        return probabilities[:, -1].mean(dim=(1, 2)).unsqueeze(1)


class ScaleOptimizeTest(unittest.TestCase):
    def optimize(self, latent: torch.Tensor | None = None, **overrides):
        kwargs = {
            "steps": 2,
            "batch_size": 3,
            "lr": 0.1,
            "t_min": 1,
            "t_max": 3,
            "num_phases": 2,
            "sds_weight": 0.0,
            "continuity_weight": 0.0,
            "preservation_weight": 0.0,
        }
        kwargs.update(overrides)
        return optimize_large_latent(
            torch.zeros(1, 4, 4, 4) if latent is None else latent,
            kwargs.pop("vae", TinyVAE()),
            kwargs.pop("diffusion_model", ZeroNoise()),
            kwargs.pop("ddpm", DDPMProcess(timesteps=4)),
            **kwargs,
        )

    def test_returns_final_latent_and_step_history(self):
        refined, history = self.optimize()

        self.assertEqual(refined.shape, torch.Size([1, 4, 4, 4]))
        self.assertEqual(history["step"].tolist(), [1, 2])
        self.assertEqual(history["loss"].shape, torch.Size([2]))

    def test_zero_steps_returns_only_initial_lmpdd(self):
        latent = torch.randn(1, 4, 4, 4)
        refined, history = self.optimize(latent, steps=0)

        self.assertTrue(torch.equal(refined, latent))
        self.assertEqual(history["step"].numel(), 0)

    def test_pretrained_critic_guides_scale_crops_without_condition(self):
        critic = RecordingCritic()
        latent = torch.zeros(1, 4, 4, 4)
        refined, stats = self.optimize(
            latent,
            steps=1,
            batch_size=2,
            critic=critic,
            critic_weight=0.1,
        )

        self.assertIn("critic", stats)
        self.assertEqual(critic.batch_sizes, [2])
        self.assertEqual(critic.channel_sizes, [2])
        self.assertFalse(torch.equal(refined, latent))

    def test_anchor_is_soft_and_updates_shared_3d_latent(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=2,
        )

        refined, history = self.optimize(
            steps=3,
            anchors=[anchor],
            anchor_weight=1.0,
        )

        self.assertGreater(float(refined[:, 1:3, 1:3, 1:3].mean()), 0.0)
        self.assertLess(float(refined.max()), 1.0)
        self.assertIn("anchor", history)

    def test_anchor_patch_matches_final_tri_axis_decoder(self):
        latent = torch.linspace(-2.0, 2.0, steps=64).view(1, 4, 4, 4)
        cases = (
            (ContextVAE(), 2, 2, slice(1, 3)),
            (UpsampleContextVAE(), 4, 4, slice(2, 6)),
            (UpsampleContextVAE(), 8, 4, slice(0, 8)),
        )
        for vae, target_size, index, span in cases:
            full = decode_large_volume_probabilities(
                vae,
                latent,
                tile_overlap=1,
                batch_size=None,
            )[0]
            for axis in range(3):
                with self.subTest(
                    axis=axis,
                    target_size=target_size,
                    factor=vae.downsample_factor
                    if hasattr(vae, "downsample_factor")
                    else 1,
                ):
                    anchor = AnchorSlice(
                        image=np.zeros((target_size, target_size), dtype=np.uint8),
                        axis=axis,
                        index=index,
                    )
                    patch = decode_anchor_patch(
                        vae,
                        latent,
                        anchor,
                        target_size=target_size,
                        num_phases=2,
                        tile_overlap=1,
                    )
                    slices = [span, span, span]
                    slices[axis] = index
                    expected = full[(slice(None), *slices)]

                    self.assertTrue(
                        torch.allclose(patch, expected, atol=1e-6, rtol=1e-6)
                    )

    def test_anchor_patch_has_finite_latent_gradient(self):
        latent = torch.randn(1, 4, 4, 4, requires_grad=True)
        patch = decode_anchor_patch(
            TinyVAE(),
            latent,
            AnchorSlice(
                image=np.zeros((2, 2), dtype=np.uint8),
                axis=1,
                index=2,
            ),
            target_size=2,
            num_phases=2,
            tile_overlap=0,
        )

        patch[1].mean().backward()

        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

    def test_cropped_anchor_patch_matches_final_decoder_region(self):
        latent = torch.linspace(-2.0, 2.0, steps=64).view(1, 4, 4, 4)
        vae = UpsampleContextVAE()
        full = decode_large_volume_probabilities(
            vae,
            latent,
            tile_overlap=1,
            batch_size=None,
        )[0]
        anchor = AnchorSlice(
            image=np.zeros((4, 4), dtype=np.uint8),
            axis=0,
            index=4,
        )

        patch = decode_anchor_patch(
            vae,
            latent,
            anchor,
            target_size=4,
            num_phases=2,
            tile_overlap=1,
            crop_start=(1, 1),
            crop_size=2,
        )

        self.assertTrue(
            torch.allclose(patch, full[:, 4, 3:5, 3:5], atol=1e-6, rtol=1e-6)
        )

    def test_three_phase_guidance_never_optimizes_ordinal_phase_values(self):
        refined, _ = self.optimize(
            vae=ThreePhaseVAE(),
            num_phases=3,
            global_fraction_weight=1.0,
            fraction_targets=torch.tensor([0.2, 0.3, 0.5]),
        )

        self.assertEqual(refined.shape, torch.Size([1, 4, 4, 4]))

    def test_global_fraction_tracks_categorical_labels_not_soft_mass(self):
        _, stats = self.optimize(
            steps=1,
            fraction_targets=torch.tensor([1.0, 0.0]),
            global_fraction_weight=1.0,
        )

        self.assertLess(float(stats["global_fraction"][-1]), 1e-5)

    def test_global_fraction_updates_the_shared_latent(self):
        refined, _ = self.optimize(
            steps=3,
            fraction_targets=torch.tensor([0.0, 1.0]),
            global_fraction_weight=1.0,
        )

        self.assertGreater(float(refined.mean()), 0.0)

    def test_sds_samples_all_axes_in_one_balanced_batch(self):
        model = RecordingNoise()
        refined, stats = self.optimize(
            steps=1,
            diffusion_model=model,
            sds_weight=1.0,
        )

        self.assertEqual(model.batch_sizes, [3])
        self.assertEqual(refined.shape, torch.Size([1, 4, 4, 4]))
        self.assertIn("sds", stats)

    def test_rejects_invalid_latent_and_anchor_contracts(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            self.optimize(torch.zeros(1, 4, 4, 4, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.optimize(torch.full((1, 4, 4, 4), float("nan")))
        with self.assertRaisesRegex(ValueError, "anchors are required"):
            self.optimize(anchor_weight=1.0)
        with self.assertRaisesRegex(ValueError, "decode_batch_size"):
            self.optimize(decode_batch_size=0)


if __name__ == "__main__":
    unittest.main()
