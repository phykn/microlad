import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.predict import AnchorSlice, PredictOptions, Predictor


class IdentityDDPM:
    def __init__(self, timesteps: int = 4) -> None:
        self.num_timesteps = timesteps
        self.steps: list[int] = []

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.steps.append(int(t[0].item()))
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_start


class ZeroDenoiser(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class IdentityVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 2
        self.latent_size = 2
        self.latent_ch = 1
        self.downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clone()


class PredictOptionsTest(unittest.TestCase):
    def test_predict_options_rejects_weights_outside_zero_to_one(self):
        with self.assertRaisesRegex(ValueError, "sds_weight"):
            PredictOptions(num_phases=2, sds_weight=1.1)
        with self.assertRaisesRegex(ValueError, "anchor_weight"):
            PredictOptions(num_phases=2, anchor_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "vf_weight"):
            PredictOptions(num_phases=2, vf_weight=2.0)


class PredictorTest(unittest.TestCase):
    def test_predict_returns_quantized_phase_volume(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options=options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_predict_accepts_options_as_first_argument(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2)

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(options)

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIsInstance(stats, dict)

    def test_predict_builds_targets_and_runs_sds_when_enabled(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(
            num_phases=2,
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            vf_weight=1.0,
        )
        target_images = [np.zeros((2, 2), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                target_images=target_images,
                options=options,
            )

        self.assertEqual(volume.shape, torch.Size([2, 2, 2]))
        self.assertIn("vf", stats)
        self.assertIn("loss", stats)

    def test_predict_blends_anchor_latent_without_forced_overwrite(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(2, 1, 2, 2)):
            volume, stats = predictor.predict(
                anchors=[anchor],
                options=PredictOptions(num_phases=2),
            )

        self.assertTrue(torch.equal(volume[1], torch.ones(2, 2, dtype=torch.uint8)))
        self.assertTrue(torch.equal(volume[0], torch.zeros(2, 2, dtype=torch.uint8)))
        self.assertIsInstance(stats, dict)

    def test_predict_rejects_target_loss_without_target_images(self):
        predictor = Predictor(IdentityVAE(), ZeroDenoiser(), IdentityDDPM(), device="cpu")
        options = PredictOptions(num_phases=2, sds_steps=1, vf_weight=1.0)

        with self.assertRaisesRegex(ValueError, "target_images"):
            predictor.predict(options)

    def test_predict_uses_volume_size_for_large_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2),
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertIsInstance(stats, dict)

    def test_predict_uses_anchor_size_without_overwriting_large_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, _ = predictor.predict(
                PredictOptions(num_phases=2),
                anchors=[anchor],
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))

    def test_predict_scale_up_accepts_vae_size_anchor_and_larger_volume(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2),
                anchors=[anchor],
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertEqual(stats["condition_start"], 1)
        self.assertTrue(torch.equal(volume[2, 1:3, 1:3], torch.ones(2, 2, dtype=torch.uint8)))

    def test_predict_refines_large_volume_when_enabled(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=1),
            device="cpu",
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(num_phases=2, refine_steps=1),
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIsInstance(stats, dict)

    def test_predict_runs_sds_for_large_volume_anchor(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((4, 4), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(
                    num_phases=2,
                    sds_steps=1,
                    sds_slice_steps=1,
                    sds_weight=0.0,
                    anchor_weight=1.0,
                ),
                anchors=[anchor],
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("steps", stats)

    def test_predict_scale_sds_visits_shifted_vae_size_anchor_slice(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                PredictOptions(
                    num_phases=2,
                    sds_steps=1,
                    sds_slice_steps=1,
                    sds_weight=0.0,
                    anchor_weight=1.0,
                ),
                anchors=[anchor],
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("anchor", stats)

    def test_predict_runs_scale_sds_with_large_target_images(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        options = PredictOptions(
            num_phases=2,
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            vf_weight=1.0,
            tpc_weight=1.0,
        )
        target_images = [np.zeros((4, 4), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                options,
                target_images=target_images,
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("vf", stats)
        self.assertIn("tpc", stats)

    def test_predict_runs_scale_sds_with_vae_size_tpc_targets(self):
        predictor = Predictor(
            IdentityVAE(),
            ZeroDenoiser(),
            IdentityDDPM(timesteps=4),
            device="cpu",
        )
        options = PredictOptions(
            num_phases=2,
            sds_steps=1,
            sds_slice_steps=1,
            sds_t_min=1,
            sds_t_max=3,
            sds_weight=0.0,
            tpc_weight=1.0,
        )
        target_images = [np.zeros((2, 2), dtype=np.uint8)]

        with patch("torch.randn", return_value=torch.zeros(1, 4, 4, 4)):
            volume, stats = predictor.predict(
                options,
                target_images=target_images,
                volume_size=4,
            )

        self.assertEqual(volume.shape, torch.Size([4, 4, 4]))
        self.assertIn("tpc", stats)


if __name__ == "__main__":
    unittest.main()
