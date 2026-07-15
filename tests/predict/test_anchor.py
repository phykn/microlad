import unittest

import numpy as np
import torch

from src.predict.anchor import (
    AnchorSlice,
    build_constraints,
    prepare_anchors,
)


class AnchorTest(unittest.TestCase):
    def test_prepares_segmented_anchor(self):
        anchor = AnchorSlice(
            np.array([[0, 0, 120, 120, 255, 255]] * 6, dtype=np.uint8),
            axis=0,
            index=3,
        )

        prepared = prepare_anchors(
            [anchor],
            volume_size=6,
            num_phases=3,
            segment=True,
            device=torch.device("cpu"),
        )

        self.assertTrue(
            torch.equal(torch.unique(prepared[0].image), torch.tensor([0, 1, 2]))
        )

    def test_rejects_invalid_phase_values(self):
        invalid_images = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[np.nan]], dtype=np.float32),
            np.array([[2]], dtype=np.uint8),
        )

        for image in invalid_images:
            with self.subTest(image=image), self.assertRaises(ValueError):
                prepare_anchors(
                    [AnchorSlice(image, axis=0, index=0)],
                    volume_size=2,
                    num_phases=2,
                    segment=False,
                    device=torch.device("cpu"),
                )

    def test_centers_smaller_anchor_in_constraint_volume(self):
        anchor = AnchorSlice(np.ones((2, 2), dtype=np.uint8), axis=1, index=2)
        prepared = prepare_anchors(
            [anchor],
            volume_size=4,
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
        )

        labels, mask = build_constraints(
            (4, 4, 4),
            prepared,
            device=torch.device("cpu"),
        )

        expected_mask = torch.zeros((4, 4, 4), dtype=torch.bool)
        expected_mask[1:3, 2, 1:3] = True
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertTrue(torch.equal(labels[mask], torch.ones(4, dtype=torch.long)))

    def test_rejects_duplicate_plane(self):
        anchors = [
            AnchorSlice(np.zeros((2, 2), dtype=np.uint8), axis=0, index=1),
            AnchorSlice(np.ones((2, 2), dtype=np.uint8), axis=0, index=1),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_anchors(
                anchors,
                volume_size=2,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )

    def test_rejects_conflicting_cross_axis_intersection(self):
        anchors = [
            AnchorSlice(np.zeros((2, 2), dtype=np.uint8), axis=0, index=1),
            AnchorSlice(
                np.array([[0, 0], [1, 1]], dtype=np.uint8),
                axis=1,
                index=0,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "Conflicting anchor intersection"):
            prepare_anchors(
                anchors,
                volume_size=2,
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
