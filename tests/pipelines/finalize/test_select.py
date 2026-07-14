import unittest

import torch

from src.app.api import QualityConfig, RefineConfig
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.finalize.select import (
    _morphology_rank,
    _violation_rank,
    select_latent_volume,
    select_probability_volume,
)


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
    def test_passing_candidate_rank_prefers_morphology_over_anchor_tie_break(self):
        zero = torch.tensor(0.0)
        rough = {
            "anchor_max_mismatch": torch.tensor(0.01),
            "phase_fraction_error": torch.zeros(2),
            "axis_exact_repeat_rate": torch.zeros(3),
            "axis_global_boundary_jump": torch.full((3,), 0.07),
            "axis_transition_rate": torch.zeros(3),
            "axis_run_profile_mae": torch.full((3,), 0.08),
        }
        natural = {
            **rough,
            "anchor_max_mismatch": torch.tensor(0.02),
            "axis_global_boundary_jump": torch.full((3,), 0.03),
            "axis_run_profile_mae": torch.full((3,), 0.04),
        }

        self.assertLess(
            _morphology_rank(natural, zero, None),
            _morphology_rank(rough, zero, None),
        )

    def test_failed_candidate_rank_keeps_anchor_priority(self):
        zero = torch.tensor(0.0)
        stats = {
            "anchor_max_mismatch": zero,
            "axis_exact_repeat_rate": torch.zeros(3),
            "axis_global_boundary_jump": torch.zeros(3),
            "axis_transition_rate": torch.zeros(3),
        }
        first = {
            "anchor": torch.tensor(0.05),
            "fraction": zero,
            "calibration": zero,
            "calibration_budget": zero,
            "repetition": zero,
            "boundary": zero,
            "transition": zero,
            "run": zero,
        }
        second = dict(first)
        second["anchor"] = torch.tensor(0.01)
        second["boundary"] = torch.tensor(0.05)

        self.assertLess(
            _violation_rank(second, stats, zero, None),
            _violation_rank(first, stats, zero, None),
        )

    def test_probability_selection_evaluates_scale_and_refine_candidates(self):
        zero = torch.zeros(2, 2, 2, dtype=torch.long)
        one = torch.ones(2, 2, 2, dtype=torch.long)
        probabilities = [
            torch.nn.functional.one_hot(value, num_classes=2)
            .movedim(-1, 0)
            .unsqueeze(0)
            .float()
            for value in (zero, one)
        ]
        volume, stats = select_probability_volume(
            probabilities,
            candidate_steps=[0, 1],
            refine_steps=[0, 0],
            num_phases=2,
            target_fraction=torch.tensor([0.0, 1.0]),
            phase_fraction_tolerance=0.0,
            anchors=[],
            references=None,
            quality=QualityConfig(
                anchor_tolerance=1.0,
                morphology_tolerance=1.0,
                continuity_tolerance=1.0,
                repeat_tolerance=1.0,
                calibration_budget=1.0,
            ),
        )

        self.assertTrue(torch.all(volume == 1))
        self.assertEqual(int(stats["selected_scale_step"]), 1)
        self.assertEqual(int(stats["selected_refine_steps"]), 0)
        self.assertEqual(stats["candidate_scale_steps"].tolist(), [0, 1])

    def test_gate_returns_best_candidate_when_calibration_exceeds_budget(self):
        volume, stats = select_latent_volume(
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
                anchor_tolerance=1.0,
                morphology_tolerance=1.0,
                continuity_tolerance=1.0,
                repeat_tolerance=1.0,
                calibration_budget=0.0,
            ),
        )

        self.assertTrue(torch.all(volume == 1))
        self.assertFalse(bool(stats["quality_passed"]))
        self.assertGreater(float(stats["quality_calibration_budget"]), 0.0)
        self.assertEqual(stats["candidate_latent_steps"].tolist(), [0])
        self.assertEqual(stats["candidate_refine_steps"].tolist(), [0])
        self.assertEqual(
            stats["candidate_calibration_changed_fractions"].shape,
            torch.Size([1]),
        )
        self.assertEqual(
            stats["candidate_refined_phase_fraction"].shape,
            torch.Size([1, 2]),
        )
        self.assertEqual(
            stats["candidate_final_axis_global_boundary_jump"].shape,
            torch.Size([1, 3]),
        )

    def test_equal_quality_prefers_latent_closest_to_lmpdd_base(self):
        _, stats = select_latent_volume(
            ConstantVAE(),
            [torch.zeros(1, 2, 2, 2), torch.ones(1, 2, 2, 2)],
            candidate_steps=[0, 1],
            num_phases=2,
            target_fraction=torch.tensor([1.0, 0.0]),
            phase_fraction_tolerance=0.0,
            anchors=[],
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

        self.assertEqual(int(stats["selected_latent_step"]), 0)
        self.assertEqual(
            stats["candidate_latent_delta_over_base_std"].shape,
            torch.Size([2]),
        )
        self.assertEqual(float(stats["selected_latent_delta_over_base_std"]), 0.0)

    def test_calibration_protects_model_labels_inside_anchor_footprint(self):
        anchor = VolumeAnchor(
            image=torch.ones(2, 2),
            axis=0,
            index=1,
        )

        volume, stats = select_latent_volume(
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
        self.assertEqual(
            stats["candidate_decoded_anchor_mismatches"].shape,
            torch.Size([1, 1]),
        )
        self.assertEqual(
            stats["candidate_refined_anchor_mismatches"].shape,
            torch.Size([1, 1]),
        )
        self.assertEqual(
            stats["candidate_final_anchor_mismatches"].shape,
            torch.Size([1, 1]),
        )
        self.assertTrue(torch.equal(stats["final_phase_fraction"], torch.tensor([0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
