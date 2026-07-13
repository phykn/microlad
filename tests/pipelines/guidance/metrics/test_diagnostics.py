import unittest

import torch

from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume


class GuidanceDiagnosticsTest(unittest.TestCase):
    def test_anchor_diagnostics_report_each_phase(self):
        volume = torch.zeros(2, 2, 2)
        volume[0, 0, 1] = 1
        anchor = VolumeAnchor(
            image=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
            axis=0,
            index=0,
        )

        stats = evaluate_phase_volume(
            volume,
            num_phases=2,
            anchors=[anchor],
        )

        self.assertTrue(
            torch.equal(
                stats["anchor_phase_mismatches"],
                torch.tensor([[0.0, 2.0 / 3.0]]),
            )
        )
        self.assertAlmostEqual(
            float(stats["anchor_max_phase_mismatch"]),
            2.0 / 3.0,
        )

    def test_fractional_phase_labels_are_rejected(self):
        volume = torch.zeros(4, 4, 4)
        volume[0, 0, 0] = 0.5

        with self.assertRaisesRegex(ValueError, "integer phase values"):
            evaluate_phase_volume(volume, num_phases=2)


if __name__ == "__main__":
    unittest.main()
