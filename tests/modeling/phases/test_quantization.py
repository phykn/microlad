import unittest

import torch

from src.modeling.phases.quantization import quantize_phase


class PredictPostprocessTest(unittest.TestCase):
    def test_quantize_phase_rounds_phase_index_values(self):
        values = torch.tensor([[0.0, 1.0, 1.8]])

        phase = quantize_phase(values, num_phases=3)

        self.assertEqual(phase.dtype, torch.uint8)
        self.assertTrue(torch.equal(phase, torch.tensor([[0, 1, 2]], dtype=torch.uint8)))

    def test_quantize_phase_clamps_values_outside_training_range(self):
        values = torch.tensor([[-1.0, 4.0]])

        phase = quantize_phase(values, num_phases=4)

        self.assertTrue(torch.equal(phase, torch.tensor([[0, 3]], dtype=torch.uint8)))

    def test_quantize_phase_rejects_invalid_num_phases(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            quantize_phase(torch.zeros(1), num_phases=1)

        with self.assertRaisesRegex(ValueError, "num_phases"):
            quantize_phase(torch.zeros(1), num_phases=2.5)

    def test_quantize_phase_rejects_num_phases_that_exceed_uint8_range(self):
        with self.assertRaisesRegex(ValueError, "num_phases"):
            quantize_phase(torch.zeros(1), num_phases=257)

    def test_quantize_phase_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_phase(torch.tensor([float("nan")]), num_phases=3)

    def test_quantize_phase_rejects_non_floating_values(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            quantize_phase(torch.tensor([0, 1]), num_phases=3)

if __name__ == "__main__":
    unittest.main()
