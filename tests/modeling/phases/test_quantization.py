import unittest

import numpy as np
import torch

from src.modeling.phases.quantization import (
    decode_phase,
    phase_to_numpy,
    quantize_phase,
)


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

    def test_decode_phase_removes_single_channel_dimension(self):
        output = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])

        phase = decode_phase(output, num_phases=3)

        self.assertEqual(phase.shape, torch.Size([1, 2, 2]))
        self.assertEqual(phase.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(
                phase,
                torch.tensor([[[0, 1], [2, 2]]], dtype=torch.uint8),
            )
        )

    def test_decode_phase_accepts_phase_logits(self):
        output = torch.tensor(
            [
                [
                    [[4.0, 0.0], [0.0, 0.0]],
                    [[0.0, 4.0], [0.0, 0.0]],
                    [[0.0, 0.0], [4.0, 5.0]],
                ]
            ]
        )

        phase = decode_phase(output, num_phases=3)

        self.assertEqual(phase.shape, torch.Size([1, 2, 2]))
        self.assertEqual(phase.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(
                phase,
                torch.tensor([[[0, 1], [2, 2]]], dtype=torch.uint8),
            )
        )

    def test_decode_phase_preserves_minority_probability_mass(self):
        probabilities = torch.tensor(
            [
                [
                    [[0.60, 0.60], [0.60, 0.60]],
                    [[0.25, 0.25], [0.25, 0.25]],
                    [[0.15, 0.15], [0.15, 0.15]],
                ]
            ]
        )

        phase = decode_phase(probabilities.log(), num_phases=3)
        counts = torch.bincount(phase.reshape(-1).long(), minlength=3)

        self.assertTrue(torch.equal(counts, torch.tensor([2, 1, 1])))

    def test_decode_phase_rejects_non_floating_output(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            decode_phase(
                torch.zeros(1, 1, 2, 2, dtype=torch.int64),
                num_phases=3,
            )

    def test_phase_to_numpy_returns_uint8_array_on_cpu(self):
        phase = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.uint8)

        array = phase_to_numpy(phase)

        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(array.shape, (1, 2, 2))
        self.assertTrue(np.array_equal(array, phase.numpy()))

    def test_phase_to_numpy_rejects_non_uint8_phase(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            phase_to_numpy(torch.zeros(1, 2, 2))


if __name__ == "__main__":
    unittest.main()
