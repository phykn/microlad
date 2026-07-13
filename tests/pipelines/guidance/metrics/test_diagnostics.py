import unittest

import torch

from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume


class GuidanceDiagnosticsTest(unittest.TestCase):
    def test_fractional_phase_labels_are_rejected(self):
        volume = torch.zeros(4, 4, 4)
        volume[0, 0, 0] = 0.5

        with self.assertRaisesRegex(ValueError, "integer phase values"):
            evaluate_phase_volume(volume, num_phases=2)


if __name__ == "__main__":
    unittest.main()
