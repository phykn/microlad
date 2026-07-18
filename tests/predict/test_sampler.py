import unittest

import torch

from src.diffusion import DDPMProcess
from src.model import encode_labels
from src.predict import ImageMPDDSampler


class RecordingDenoiser(torch.nn.Module):
    def __init__(
        self,
        num_phases: int,
    ) -> None:
        super().__init__()
        self.num_phases = num_phases
        self.conditions = []
        self.axis_conditions = []

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
        self.conditions.append(
            None if phase_fractions is None else phase_fractions.detach().clone()
        )
        self.axis_conditions.append(
            None if axis_condition is None else axis_condition.detach().clone()
        )
        return torch.zeros_like(x)


class ImageMPDDSamplerTest(unittest.TestCase):
    def test_sampling_forwards_fraction_condition_with_soft_anchor(self):
        model = RecordingDenoiser(num_phases=2)
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=3, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )
        labels = torch.zeros(1, 1, 8, 8, 8)
        labels[:, :, 2] = 1
        anchor = encode_labels(labels, 2)[0]
        mask = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        mask[:, 2] = True

        sample = sampler.sample(
            8,
            phase_fractions=(0.25, 0.75),
            anchor_image=anchor,
            anchor_mask=mask,
            harmonization_steps=2,
            batch_size=8,
        )

        self.assertFalse(torch.equal(sample[:, 2], anchor[:, 2]))
        self.assertEqual(len(model.conditions), 5)
        self.assertTrue(
            all(
                condition is not None
                and torch.allclose(
                    condition,
                    torch.tensor([[0.25, 0.75]]).expand_as(condition),
                )
                for condition in model.conditions
            )
        )

    def test_tiled_scale_up_returns_requested_volume_shape(self):
        sampler = ImageMPDDSampler(
            RecordingDenoiser(num_phases=2),
            DDPMProcess(timesteps=1, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        sample = sampler.sample(
            12,
            harmonization_steps=1,
            tile_overlap=4,
            batch_size=6,
        )

        self.assertEqual(sample.shape, torch.Size([2, 12, 12, 12]))

    def test_ddpm_forwards_axis_sequence_without_fraction_condition(self):
        model = RecordingDenoiser(num_phases=2)
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=4, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        sampler.sample(
            8,
            harmonization_steps=1,
            batch_size=8,
            progress=False,
        )

        self.assertTrue(all(condition is None for condition in model.conditions))
        self.assertEqual(
            [int(axis.unique().item()) for axis in model.axis_conditions],
            [0, 1, 2, 0],
        )
        self.assertTrue(
            all(axis.dtype == torch.long for axis in model.axis_conditions)
        )

    def test_ddim_forwards_axis_sequence(self):
        model = RecordingDenoiser(num_phases=2)
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=6, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        sampler.sample(
            8,
            phase_fractions=(0.4, 0.6),
            harmonization_steps=1,
            batch_size=8,
            ddim_steps=3,
            progress=False,
        )

        self.assertEqual(
            [int(axis.unique().item()) for axis in model.axis_conditions],
            [0, 1, 2],
        )

    def test_tiled_harmonization_keeps_each_axis_condition(self):
        model = RecordingDenoiser(num_phases=2)
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=3, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        sampler.sample(
            12,
            harmonization_steps=2,
            tile_overlap=4,
            batch_size=6,
            progress=False,
        )

        recorded = [int(axis.unique().item()) for axis in model.axis_conditions]
        self.assertEqual(recorded, [0] * 16 + [1] * 16 + [2] * 8)
        self.assertTrue(
            all(axis.shape == torch.Size([6]) for axis in model.axis_conditions)
        )

    def test_ddim_harmonization_uses_soft_anchor_after_skipped_steps(self):
        model = RecordingDenoiser(num_phases=2)
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=6, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )
        labels = torch.zeros(1, 1, 8, 8, 8)
        labels[:, :, 5] = 1
        anchor = encode_labels(labels, 2)[0]
        mask = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        mask[:, 5] = True

        sample = sampler.sample(
            8,
            phase_fractions=(0.4, 0.6),
            anchor_image=anchor,
            anchor_mask=mask,
            harmonization_steps=3,
            batch_size=8,
            ddim_steps=3,
        )

        self.assertFalse(torch.equal(sample[:, 5], anchor[:, 5]))
        self.assertEqual(len(model.conditions), 7)

    def test_tiled_ddim_scale_up_returns_requested_volume_shape(self):
        sampler = ImageMPDDSampler(
            RecordingDenoiser(num_phases=2),
            DDPMProcess(timesteps=4, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        sample = sampler.sample(
            12,
            harmonization_steps=1,
            tile_overlap=4,
            batch_size=6,
            ddim_steps=2,
        )

        self.assertEqual(sample.shape, torch.Size([2, 12, 12, 12]))

    def test_guidance_requires_fraction_condition(self):
        sampler = ImageMPDDSampler(
            RecordingDenoiser(num_phases=2),
            DDPMProcess(timesteps=2, beta_start=0.01, beta_end=0.02),
            image_size=8,
            num_phases=2,
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "phase_fractions"):
            sampler.sample(8, guidance_scale=2.0)


if __name__ == "__main__":
    unittest.main()
