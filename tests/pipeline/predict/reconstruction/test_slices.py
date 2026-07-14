import unittest

import torch

from src.pipeline.predict.reconstruction.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
)


class PredictSlicesTest(unittest.TestCase):
    def test_slice_operations_require_3d_volume(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            extract_slice(torch.zeros(2, 2), axis=0, index=0)

    def test_extract_slice_rejects_invalid_axis(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "axis"):
            extract_slice(volume, axis=99, index=0)

        with self.assertRaisesRegex(ValueError, "axis.*integer"):
            extract_slice(volume, axis=True, index=0)

    def test_extract_slice_rejects_out_of_range_index(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "index"):
            extract_slice(volume, axis=0, index=2)

        with self.assertRaisesRegex(ValueError, "index.*integer"):
            extract_slice(volume, axis=0, index=1.0)

    def test_replace_slice_rejects_invalid_axis(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "axis"):
            replace_slice(volume, axis=99, index=0, image=torch.zeros(2, 3))

    def test_replace_slice_rejects_out_of_range_index(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "index"):
            replace_slice(volume, axis=1, index=3, image=torch.zeros(2, 4))

    def test_replace_slice_rejects_image_shape_mismatch(self):
        volume = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "shape"):
            replace_slice(volume, axis=1, index=0, image=torch.zeros(3, 4))

    def test_replace_slice_requires_matching_dtype(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace_slice(
                torch.zeros(2, 2, 2),
                axis=0,
                index=0,
                image=torch.zeros(2, 2, dtype=torch.float64),
            )

    def test_extract_slice_batch_keeps_batch_first_for_each_axis(self):
        volume = torch.arange(27).view(3, 3, 3)

        axis_zero = extract_slice_batch(volume, axis=0, indices=[0, 2])
        axis_one = extract_slice_batch(volume, axis=1, indices=[0, 2])
        axis_two = extract_slice_batch(volume, axis=2, indices=[0, 2])

        self.assertTrue(torch.equal(axis_zero[1], volume[2]))
        self.assertTrue(torch.equal(axis_one[1], volume[:, 2, :]))
        self.assertTrue(torch.equal(axis_two[1], volume[:, :, 2]))

    def test_replace_slice_batch_writes_each_axis(self):
        volume = torch.zeros(3, 3, 3)
        images = torch.stack(
            [
                torch.ones(3, 3),
                torch.full((3, 3), 2.0),
            ]
        )

        replace_slice_batch(volume, axis=1, indices=[0, 2], images=images)

        self.assertTrue(torch.equal(volume[:, 0, :], images[0]))
        self.assertTrue(torch.equal(volume[:, 2, :], images[1]))
        self.assertTrue(torch.equal(volume[:, 1, :], torch.zeros(3, 3)))

    def test_replace_slice_batch_rejects_image_spatial_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "image shape"):
            replace_slice_batch(
                torch.zeros(3, 3, 3),
                axis=1,
                indices=[0, 2],
                images=torch.zeros(2, 4, 3),
            )

    def test_extract_slice_batch_rejects_out_of_range_indices(self):
        with self.assertRaisesRegex(ValueError, "indices"):
            extract_slice_batch(torch.zeros(3, 3, 3), axis=0, indices=[3])

if __name__ == "__main__":
    unittest.main()
