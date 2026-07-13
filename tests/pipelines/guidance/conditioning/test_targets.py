import unittest

import numpy as np
import torch

from src.pipelines.guidance.conditioning.targets import (
    build_descriptor_targets,
    prepare_target_images,
)


class DescriptorTargetsTest(unittest.TestCase):
    def test_prepares_categorical_and_segmented_target_images(self):
        categorical = prepare_target_images(
            [np.array([[0, 1], [2, 2]], dtype=np.int64)],
            num_phases=3,
        )
        segmented = prepare_target_images(
            [np.array([[0, 0, 120, 120, 255, 255]], dtype=np.uint8)],
            num_phases=3,
            segment=True,
        )

        self.assertEqual(categorical.dtype, torch.long)
        self.assertTrue(torch.equal(categorical, torch.tensor([[[0, 1], [2, 2]]])))
        self.assertTrue(torch.equal(segmented.unique(), torch.tensor([0, 1, 2])))

    def test_builds_only_selected_descriptor_targets(self):
        labels = prepare_target_images(
            [
                np.array([[0, 1], [2, 2]], dtype=np.uint8),
                np.array([[0, 0], [1, 2]], dtype=np.uint8),
            ],
            num_phases=3,
        )

        targets = build_descriptor_targets(
            labels,
            num_phases=3,
            use_fraction=True,
        )

        self.assertEqual(set(targets), {"fraction_targets"})
        self.assertTrue(
            torch.allclose(
                targets["fraction_targets"],
                torch.tensor([3.0 / 8.0, 2.0 / 8.0, 3.0 / 8.0]),
                atol=1e-3,
            )
        )

    def test_builds_all_optional_targets(self):
        labels = prepare_target_images(
            [
                np.array(
                    [
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                    ],
                    dtype=np.uint8,
                )
            ],
            num_phases=2,
        )

        targets = build_descriptor_targets(
            labels,
            num_phases=2,
            use_fraction=True,
            use_tpc=True,
            use_sa=True,
            use_diffusivity=True,
            diffusivity_grid_size=2,
        )

        self.assertEqual(
            set(targets),
            {
                "fraction_targets",
                "tpc_targets",
                "sa_targets",
                "diffusivity_targets",
                "diffusivity_solver",
            },
        )
        self.assertEqual(targets["fraction_targets"].shape, torch.Size([2]))
        self.assertEqual(targets["tpc_targets"].shape[0], 2)
        self.assertEqual(targets["sa_targets"].shape, torch.Size([2]))
        self.assertEqual(targets["diffusivity_targets"].shape, torch.Size([2]))

    def test_no_selected_target_does_not_require_labels(self):
        self.assertEqual(build_descriptor_targets(None, num_phases=2), {})

    def test_preparation_rejects_invalid_images_once(self):
        with self.assertRaisesRegex(ValueError, "images"):
            prepare_target_images([], num_phases=2)
        with self.assertRaisesRegex(TypeError, "numpy"):
            prepare_target_images([[[0, 1]]], num_phases=2)
        with self.assertRaisesRegex(ValueError, "integer"):
            prepare_target_images(
                [np.array([[0.0, 0.5]], dtype=np.float32)],
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_target_images(
                [np.array([[0.0, np.nan]], dtype=np.float32)],
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "0 to 1"):
            prepare_target_images(
                [np.array([[0, 2]], dtype=np.uint8)],
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            prepare_target_images(
                [
                    np.zeros((2, 2), dtype=np.uint8),
                    np.zeros((3, 3), dtype=np.uint8),
                ],
                num_phases=2,
            )

    def test_build_rejects_unprepared_labels_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "target labels"):
            build_descriptor_targets(None, num_phases=2, use_fraction=True)
        with self.assertRaisesRegex(ValueError, "torch.long"):
            build_descriptor_targets(
                torch.zeros(1, 2, 2),
                num_phases=2,
                use_fraction=True,
            )
        with self.assertRaisesRegex(ValueError, "diffusivity_grid_size"):
            build_descriptor_targets(
                torch.zeros(1, 2, 2, dtype=torch.long),
                num_phases=2,
                use_diffusivity=True,
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_descriptor_targets(
                torch.zeros(1, 2, 2, dtype=torch.long),
                num_phases=2,
                use_fraction=True,
                temperature=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "sa_kernel_size.*integer"):
            build_descriptor_targets(
                torch.zeros(1, 2, 2, dtype=torch.long),
                num_phases=2,
                use_sa=True,
                sa_kernel_size=True,
            )


if __name__ == "__main__":
    unittest.main()
