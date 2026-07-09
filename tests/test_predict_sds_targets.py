import unittest

import torch

from src.guidance.target_values import phase_vector_target


class PredictSDSTargetsTest(unittest.TestCase):
    def test_phase_vector_target_rejects_non_integer_phase_keys(self):
        with self.assertRaisesRegex(ValueError, "phase indices"):
            phase_vector_target(
                {0.5: 0.5, 1: 0.5},
                num_phases=2,
                device=torch.device("cpu"),
                dtype=torch.float32,
                label="fraction",
            )

    def test_phase_vector_target_rejects_non_floating_dtype(self):
        with self.assertRaisesRegex(ValueError, "dtype.*floating"):
            phase_vector_target(
                {0: 0.5, 1: 0.5},
                num_phases=2,
                device=torch.device("cpu"),
                dtype=torch.int64,
                label="fraction",
            )


if __name__ == "__main__":
    unittest.main()
