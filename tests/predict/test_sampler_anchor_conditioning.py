import unittest

import torch

from src.diffusion import DDPMProcess
from src.predict import ImageMPDDSampler
from src.predict.noise import guide_noise, predict_tiles
from src.predict.volume import slice_volume


class RecordingAnchorDenoiser(torch.nn.Module):
    def __init__(self, *, num_axis_conditions: int = 3) -> None:
        super().__init__()
        self.num_axis_conditions = num_axis_conditions
        self.anchor_conditioning = True
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def forward(
        self,
        image,
        steps,
        phase_fractions=None,
        axis_condition=None,
        *,
        anchor_image=None,
        anchor_mask=None,
    ):
        self.calls.append(
            {
                "image": image.detach().clone(),
                "steps": steps.detach().clone(),
                "condition": (
                    None
                    if phase_fractions is None
                    else phase_fractions.detach().clone()
                ),
                "axis": (
                    None
                    if axis_condition is None
                    else axis_condition.detach().clone()
                ),
                "anchor": (
                    None if anchor_image is None else anchor_image.detach().clone()
                ),
                "mask": (
                    None if anchor_mask is None else anchor_mask.detach().clone()
                ),
            }
        )
        if phase_fractions is None:
            return torch.zeros_like(image)
        return torch.ones_like(image) * phase_fractions[:, :1, None, None]


class LegacyDenoiser(torch.nn.Module):
    def forward(self, image, steps, phase_fractions=None, axis_condition=None):
        return torch.zeros_like(image)


class AnchorConditioningNoiseTest(unittest.TestCase):
    def test_guide_noise_duplicates_anchor_mask_and_axis_for_phase_cfg(self):
        model = RecordingAnchorDenoiser()
        patch = torch.zeros(2, 2, 4, 4)
        steps = torch.tensor([7, 5], dtype=torch.long)
        condition = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        axes = torch.tensor([0, 2], dtype=torch.long)
        anchor = torch.arange(patch.numel(), dtype=torch.float32).view_as(patch)
        mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        mask[0, :, :2] = True
        mask[1, :, 2:] = True

        noise = guide_noise(
            model,
            patch,
            steps,
            condition=condition,
            axis_condition=axes,
            guidance=2.0,
            anchor_image=anchor,
            anchor_mask=mask,
        )

        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertTrue(torch.equal(call["image"], torch.cat([patch, patch])))
        self.assertTrue(torch.equal(call["steps"], torch.cat([steps, steps])))
        self.assertTrue(torch.equal(call["axis"], torch.cat([axes, axes])))
        self.assertTrue(torch.equal(call["anchor"], torch.cat([anchor, anchor])))
        self.assertTrue(torch.equal(call["mask"], torch.cat([mask, mask])))
        self.assertTrue(
            torch.equal(
                call["condition"],
                torch.cat([torch.zeros_like(condition), condition]),
            )
        )
        expected = condition[:, :1, None, None].expand_as(noise) * 2.0
        self.assertTrue(torch.equal(noise, expected))

    def test_predict_tiles_slices_batches_before_duplicating_cfg_inputs(self):
        model = RecordingAnchorDenoiser()
        planes = torch.zeros(3, 2, 4, 4)
        steps = torch.tensor([9, 8, 7], dtype=torch.long)
        anchor = torch.arange(planes.numel(), dtype=torch.float32).view_as(planes)
        mask = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
        mask[0, :, :1] = True
        mask[1, :, 1:3] = True
        mask[2, :, 3:] = True

        noise = predict_tiles(
            model,
            planes,
            steps,
            tile_size=4,
            overlap=0,
            batch_size=2,
            fractions=torch.tensor([0.2, 0.8]),
            axis_condition=1,
            guidance=1.5,
            anchor_image=anchor,
            anchor_mask=mask,
        )

        self.assertEqual(len(model.calls), 2)
        for call, start, stop in (
            (model.calls[0], 0, 2),
            (model.calls[1], 2, 3),
        ):
            batch_anchor = anchor[start:stop]
            batch_mask = mask[start:stop]
            batch_steps = steps[start:stop]
            batch_size = stop - start
            condition = torch.tensor([[0.2, 0.8]]).expand(batch_size, -1)
            self.assertTrue(
                torch.equal(call["anchor"], torch.cat([batch_anchor, batch_anchor]))
            )
            self.assertTrue(
                torch.equal(call["mask"], torch.cat([batch_mask, batch_mask]))
            )
            self.assertTrue(
                torch.equal(
                    call["axis"],
                    torch.full((batch_size * 2,), 1, dtype=torch.long),
                )
            )
            self.assertTrue(
                torch.equal(call["steps"], torch.cat([batch_steps, batch_steps]))
            )
            self.assertTrue(
                torch.equal(
                    call["condition"],
                    torch.cat([torch.zeros_like(condition), condition]),
                )
            )
        self.assertTrue(torch.allclose(noise, torch.full_like(noise, 0.3)))


class AnchorConditioningSamplerTest(unittest.TestCase):
    @staticmethod
    def _anchor(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = torch.arange(
            2 * size**3,
            dtype=torch.float32,
        ).reshape(2, size, size, size)
        anchor = anchor / anchor.max().clamp_min(1.0)
        coordinates = torch.arange(size**3).reshape(size, size, size)
        mask = (coordinates.remainder(3) == 0).unsqueeze(0)
        return anchor, mask

    def test_anchor_aware_sampler_passes_plane_slices_without_raw_injection(self):
        size = 3
        model = RecordingAnchorDenoiser()
        sampler = ImageMPDDSampler(
            model,
            DDPMProcess(timesteps=3, beta_start=0.01, beta_end=0.02),
            image_size=size,
            num_phases=2,
            device="cpu",
        )
        anchor, mask = self._anchor(size)

        self.assertFalse(hasattr(sampler, "_inject_anchor"))
        torch.manual_seed(17)
        anchored = sampler.sample(
            size,
            phase_fractions=(0.4, 0.6),
            anchor_image=anchor,
            anchor_mask=mask,
            harmonization_steps=1,
            batch_size=size,
            progress=False,
        )
        anchored_calls = list(model.calls)
        self.assertEqual(anchored.shape, torch.Size((2, size, size, size)))
        self.assertEqual(len(anchored_calls), 3)
        for axis, call in enumerate(anchored_calls):
            clean_plane = slice_volume(anchor, axis)
            self.assertTrue(torch.equal(call["anchor"], clean_plane))
            self.assertTrue(torch.equal(call["mask"], slice_volume(mask, axis)))
            self.assertTrue(
                torch.equal(
                    call["axis"],
                    torch.full((size,), axis, dtype=torch.long),
                )
            )

    def test_legacy_sampler_rejects_anchor_inputs(self):
        size = 3
        sampler = ImageMPDDSampler(
            LegacyDenoiser(),
            DDPMProcess(timesteps=3, beta_start=0.01, beta_end=0.02),
            image_size=size,
            num_phases=2,
            device="cpu",
        )
        anchor, mask = self._anchor(size)

        with self.assertRaisesRegex(ValueError, "anchor_conditioning"):
            sampler.sample(
                size,
                anchor_image=anchor,
                anchor_mask=mask,
                harmonization_steps=1,
                batch_size=size,
                progress=False,
            )

if __name__ == "__main__":
    unittest.main()
