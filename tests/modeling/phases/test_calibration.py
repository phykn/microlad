import unittest

import torch

from src.modeling.phases.calibration import probabilities_to_calibrated_labels


class PhaseCalibrationTest(unittest.TestCase):
    def test_calibrated_labels_preserve_probability_mass_counts(self):
        probabilities = torch.tensor(
            [
                [
                    [[0.60, 0.60], [0.60, 0.60]],
                    [[0.25, 0.25], [0.25, 0.25]],
                    [[0.15, 0.15], [0.15, 0.15]],
                ]
            ]
        )

        labels = probabilities_to_calibrated_labels(probabilities, 3)
        counts = torch.bincount(labels.reshape(-1), minlength=3)

        self.assertEqual(labels.shape, torch.Size([1, 1, 2, 2]))
        self.assertTrue(torch.equal(counts, torch.tensor([2, 1, 1])))

    def test_calibrated_labels_use_lowest_probability_cost(self):
        probabilities = torch.tensor(
            [
                [
                    [[0.51, 0.90]],
                    [[0.49, 0.10]],
                ]
            ]
        )

        labels = probabilities_to_calibrated_labels(probabilities, 2)

        self.assertTrue(torch.equal(labels[0, 0], torch.tensor([[1, 0]])))

    def test_calibrated_labels_can_match_condition_phase_fractions(self):
        probabilities = torch.full((1, 3, 2, 5), 1.0 / 3.0)

        labels = probabilities_to_calibrated_labels(
            probabilities,
            3,
            target_fractions=torch.tensor([0.5, 0.3, 0.2]),
        )
        counts = torch.bincount(labels.reshape(-1), minlength=3)

        self.assertTrue(torch.equal(counts, torch.tensor([5, 3, 2])))

    def test_calibrated_labels_reject_invalid_condition_fractions(self):
        probabilities = torch.full((1, 3, 2, 2), 1.0 / 3.0)

        with self.assertRaisesRegex(ValueError, "sum to one"):
            probabilities_to_calibrated_labels(
                probabilities,
                3,
                target_fractions=torch.tensor([0.5, 0.5, 0.5]),
            )

    def test_calibrated_labels_preserve_fixed_anchor_and_total_counts(self):
        probabilities = torch.tensor(
            [[[[0.9, 0.9], [0.9, 0.9]], [[0.1, 0.1], [0.1, 0.1]]]]
        )
        fixed_labels = torch.tensor([[[[1, 0], [0, 0]]]])
        fixed_mask = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.bool)

        labels = probabilities_to_calibrated_labels(
            probabilities,
            2,
            target_fractions=torch.tensor([0.5, 0.5]),
            fixed_labels=fixed_labels,
            fixed_mask=fixed_mask,
        )
        counts = torch.bincount(labels.reshape(-1), minlength=2)

        self.assertEqual(int(labels[0, 0, 0, 0]), 1)
        self.assertTrue(torch.equal(counts, torch.tensor([2, 2])))

    def test_calibrated_labels_reject_impossible_fixed_counts(self):
        probabilities = torch.full((1, 2, 2, 2), 0.5)
        fixed_labels = torch.ones(1, 1, 2, 2)
        fixed_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "exceed"):
            probabilities_to_calibrated_labels(
                probabilities,
                2,
                target_fractions=torch.tensor([0.5, 0.5]),
                fixed_labels=fixed_labels,
                fixed_mask=fixed_mask,
            )

    def test_calibrated_labels_prioritize_fixed_anchor_without_fraction_target(self):
        probabilities = torch.tensor(
            [[[[0.99, 0.99], [0.99, 0.99]], [[0.01, 0.01], [0.01, 0.01]]]]
        )
        fixed_labels = torch.tensor([[[[1, 1], [0, 0]]]])
        fixed_mask = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)

        labels = probabilities_to_calibrated_labels(
            probabilities,
            2,
            fixed_labels=fixed_labels,
            fixed_mask=fixed_mask,
        )

        self.assertTrue(torch.all(labels[0, 0, 0] == 1))


if __name__ == "__main__":
    unittest.main()
