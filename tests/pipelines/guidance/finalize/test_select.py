import unittest

import torch

from src.app.api import QualityConfig, RefineConfig
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.finalize.select import select_volume


class ConstantVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2
    downsample_factor = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(image), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        phase_zero = torch.full_like(latent, 0.9)
        return torch.cat([phase_zero, 1.0 - phase_zero], dim=1)


class FinalSelectionTest(unittest.TestCase):
    def test_strict_gate_rejects_calibration_over_budget(self):
        with self.assertRaisesRegex(RuntimeError, "calibration_budget"):
            select_volume(
                ConstantVAE(),
                [torch.zeros(1, 2, 2, 2)],
                candidate_steps=[0],
                num_phases=2,
                target_fraction=torch.tensor([0.0, 1.0]),
                phase_fraction_tolerance=0.0,
                anchors=[],
                references=None,
                refine=RefineConfig(candidates=(0,)),
                quality=QualityConfig(
                    strict=True,
                    anchor_tolerance=1.0,
                    morphology_tolerance=1.0,
                    continuity_tolerance=1.0,
                    repeat_tolerance=1.0,
                    calibration_budget=0.0,
                ),
            )

    def test_calibration_protects_model_labels_inside_anchor_footprint(self):
        anchor = VolumeAnchor(
            image=torch.ones(2, 2),
            axis=0,
            index=1,
        )

        volume, stats = select_volume(
            ConstantVAE(),
            [torch.zeros(1, 2, 2, 2)],
            candidate_steps=[0],
            num_phases=2,
            target_fraction=torch.tensor([0.5, 0.5]),
            phase_fraction_tolerance=0.0,
            anchors=[anchor],
            references=None,
            refine=RefineConfig(candidates=(0,)),
            quality=QualityConfig(
                anchor_tolerance=1.0,
                morphology_tolerance=1.0,
                continuity_tolerance=1.0,
                repeat_tolerance=1.0,
                calibration_budget=1.0,
            ),
        )

        self.assertTrue(torch.all(volume[1] == 0))
        self.assertTrue(torch.equal(
            stats["calibration_anchor_delta"],
            torch.zeros_like(stats["calibration_anchor_delta"]),
        ))
        self.assertTrue(torch.equal(stats["final_phase_fraction"], torch.tensor([0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
