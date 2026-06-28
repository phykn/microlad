import unittest

import numpy as np
import torch

from src.predict import AnchorSlice
from src.predict.anchor import prepare_anchor_image, validate_anchor, validate_anchors


class PredictAnchorTest(unittest.TestCase):
    def test_prepare_anchor_image_scales_phase_image_to_tensor(self):
        image = np.array([[0, 1, 2]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 3]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(tensor[0, 0, 0], torch.tensor([-1.0, 0.0, 1.0]))
        )

    def test_prepare_anchor_image_can_segment_grayscale_image(self):
        image = np.array([[0, 0, 120, 120, 255, 255]], dtype=np.uint8)

        tensor = prepare_anchor_image(image, num_phases=3, segment=True)

        self.assertEqual(tensor.shape, torch.Size([1, 1, 1, 6]))
        self.assertTrue(torch.equal(torch.unique(tensor), torch.tensor([-1.0, 0.0, 1.0])))

    def test_prepare_anchor_image_rejects_non_2d_image(self):
        image = np.zeros((1, 4, 4), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "2D"):
            prepare_anchor_image(image, num_phases=2)

    def test_prepare_anchor_image_rejects_invalid_phase_values(self):
        image = np.array([[0, 2]], dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "0 to 1"):
            prepare_anchor_image(image, num_phases=2)

    def test_validate_anchor_accepts_matching_slice_shape(self):
        anchor = AnchorSlice(image=np.zeros((5, 6), dtype=np.uint8), axis=0, index=2)

        validate_anchor(anchor, volume_shape=(4, 5, 6))

    def test_validate_anchor_rejects_shape_mismatch(self):
        anchor = AnchorSlice(image=np.zeros((4, 6), dtype=np.uint8), axis=0, index=0)

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_anchor(anchor, volume_shape=(4, 5, 6))

    def test_validate_anchors_rejects_duplicate_axis_index(self):
        anchors = [
            AnchorSlice(image=np.zeros((5, 6), dtype=np.uint8), axis=0, index=2),
            AnchorSlice(image=np.ones((5, 6), dtype=np.uint8), axis=0, index=2),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_anchors(anchors, volume_shape=(4, 5, 6))


if __name__ == "__main__":
    unittest.main()
