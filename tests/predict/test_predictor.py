import unittest

import numpy as np
import torch

from src.predict import AnchorSlice, MPDDOptions, MPDDPredictor
from src.diffusion import DDPMProcess


class ZeroDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor_conditioning = True

    def forward(
        self,
        x,
        t,
        phase_fractions=None,
        axis_condition=None,
        *,
        anchor_image=None,
        anchor_mask=None,
    ):
        return torch.zeros_like(x)


class MPDDOptionsTest(unittest.TestCase):
    def test_normalizes_fraction_sequence(self):
        options = MPDDOptions(num_phases=2, phase_fractions=[0.25, 0.75])

        self.assertEqual(options.phase_fractions, (0.25, 0.75))

    def test_validates_ddim_steps(self):
        self.assertEqual(MPDDOptions(num_phases=2, ddim_steps=50).ddim_steps, 50)
        with self.assertRaisesRegex(ValueError, "ddim_steps"):
            MPDDOptions(num_phases=2, ddim_steps=0)

    def test_guidance_requires_fraction_condition(self):
        options = MPDDOptions(
            num_phases=2,
            phase_fractions=(0.4, 0.6),
            guidance_scale=2.0,
        )

        self.assertEqual(options.guidance_scale, 2.0)
        with self.assertRaisesRegex(ValueError, "phase_fractions"):
            MPDDOptions(num_phases=2, guidance_scale=2.0)

class MPDDPredictorTest(unittest.TestCase):
    def _predictor(self, model=None):
        return MPDDPredictor(
            ZeroDenoiser() if model is None else model,
            DDPMProcess(timesteps=1, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

    def test_predict_uses_fraction_only_as_a_denoiser_condition(self):
        anchor_image = np.zeros((8, 8), dtype=np.uint8)
        anchor_image[:, 4:] = 1
        options = MPDDOptions(
            num_phases=2,
            volume_size=8,
            phase_fractions=(0.25, 0.75),
            harmonization_steps=1,
            batch_size=8,
            progress=False,
        )

        torch.manual_seed(7)
        volume, stats = self._predictor().predict(
            options,
            anchors=[AnchorSlice(anchor_image, axis=0, index=3)],
        )
        unconditional = MPDDOptions(
            num_phases=2,
            volume_size=8,
            harmonization_steps=1,
            batch_size=8,
            progress=False,
        )
        torch.manual_seed(7)
        without_condition, _ = self._predictor().predict(unconditional)

        self.assertTrue(torch.equal(volume, without_condition))
        self.assertEqual(stats["anchor_voxels"], 64)

    def test_predict_reports_ddim_sampling(self):
        options = MPDDOptions(
            num_phases=2,
            volume_size=8,
            harmonization_steps=2,
            ddim_steps=1,
            progress=False,
        )

        volume, stats = self._predictor().predict(options)

        self.assertEqual(volume.shape, torch.Size([8, 8, 8]))
        self.assertEqual(stats["sampler"], "ddim")
        self.assertEqual(stats["sampling_steps"], 1)

    def test_predict_reports_fraction_guidance_scale(self):
        options = MPDDOptions(
            num_phases=2,
            volume_size=8,
            phase_fractions=(0.4, 0.6),
            harmonization_steps=1,
            ddim_steps=1,
            guidance_scale=2.0,
            progress=False,
        )

        _, stats = self._predictor().predict(options)

        self.assertEqual(stats["guidance_scale"], 2.0)

    def test_soft_anchor_does_not_create_a_hard_fraction_conflict(self):
        options = MPDDOptions(
            num_phases=2,
            volume_size=8,
            phase_fractions=(0.0, 1.0),
            progress=False,
        )

        volume, stats = self._predictor().predict(
            options,
            anchors=[
                AnchorSlice(
                    np.zeros((8, 8), dtype=np.uint8),
                    axis=0,
                    index=3,
                )
            ],
        )

        self.assertEqual(volume.shape, torch.Size([8, 8, 8]))
        self.assertEqual(stats["anchor_voxels"], 64)

    def test_scale_up_centers_smaller_anchor_in_requested_volume(self):
        anchor_image = np.zeros((8, 8), dtype=np.uint8)
        anchor_image[2:6, 2:6] = 1
        options = MPDDOptions(
            num_phases=2,
            volume_size=12,
            harmonization_steps=1,
            tile_overlap=0.5,
            batch_size=6,
            progress=False,
        )

        torch.manual_seed(11)
        volume, stats = self._predictor().predict(
            options,
            anchors=[AnchorSlice(anchor_image, axis=1, index=6)],
        )
        torch.manual_seed(11)
        without_anchor, _ = self._predictor().predict(options)

        self.assertTrue(torch.equal(volume, without_anchor))
        self.assertEqual(stats["anchor_voxels"], 64)


if __name__ == "__main__":
    unittest.main()
