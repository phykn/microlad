import unittest

import numpy as np
import torch

from src.predict.targets import build_sds_targets


class PredictTargetsTest(unittest.TestCase):
    def test_build_sds_targets_returns_only_selected_target_args(self):
        images = [
            np.array([[0, 1], [2, 2]], dtype=np.uint8),
            np.array([[0, 0], [1, 2]], dtype=np.uint8),
        ]

        targets = build_sds_targets(
            images,
            num_phases=3,
            use_vf=True,
        )

        self.assertEqual(set(targets), {"vf_targets"})
        self.assertTrue(
            torch.allclose(
                targets["vf_targets"],
                torch.tensor([3.0 / 8.0, 2.0 / 8.0, 3.0 / 8.0]),
                atol=1e-3,
            )
        )

    def test_build_sds_targets_can_segment_grayscale_images(self):
        image = np.array([[0, 0, 120, 120, 255, 255]], dtype=np.uint8)

        targets = build_sds_targets(
            [image],
            num_phases=3,
            segment=True,
            use_vf=True,
        )

        self.assertTrue(
            torch.allclose(
                targets["vf_targets"],
                torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                atol=1e-3,
            )
        )

    def test_build_sds_targets_can_build_all_optional_targets(self):
        image = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )

        targets = build_sds_targets(
            [image],
            num_phases=2,
            use_vf=True,
            use_tpc=True,
            use_sa=True,
            use_diffusivity=True,
            diffusivity_size=2,
        )

        self.assertEqual(
            set(targets),
            {
                "vf_targets",
                "tpc_targets",
                "sa_targets",
                "diffusivity_targets",
                "diffusivity_solver",
            },
        )
        self.assertEqual(targets["vf_targets"].shape, torch.Size([2]))
        self.assertEqual(targets["tpc_targets"].shape[0], 2)
        self.assertEqual(targets["sa_targets"].shape, torch.Size([2]))
        self.assertEqual(targets["diffusivity_targets"].shape, torch.Size([2]))
        self.assertEqual(targets["diffusivity_solver"].height, 2)
        self.assertEqual(targets["diffusivity_solver"].width, 2)

    def test_build_sds_targets_returns_empty_args_when_no_target_is_selected(self):
        targets = build_sds_targets([], num_phases=2)

        self.assertEqual(targets, {})

    def test_build_sds_targets_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "images"):
            build_sds_targets([], num_phases=2, use_vf=True)
        with self.assertRaisesRegex(ValueError, "num_phases.*integer"):
            build_sds_targets(
                [np.array([[0, 1]], dtype=np.uint8)],
                num_phases=2.5,
                use_vf=True,
            )
        with self.assertRaisesRegex(ValueError, "num_phases"):
            build_sds_targets(
                [np.array([[0, 255]], dtype=np.uint8)],
                num_phases=257,
                use_vf=True,
            )
        with self.assertRaisesRegex(ValueError, "0 to 1"):
            build_sds_targets(
                [np.array([[0, 2]], dtype=np.uint8)],
                num_phases=2,
                use_vf=True,
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            build_sds_targets(
                [
                    np.zeros((2, 2), dtype=np.uint8),
                    np.zeros((3, 3), dtype=np.uint8),
                ],
                num_phases=2,
                use_vf=True,
            )
        with self.assertRaisesRegex(ValueError, "diffusivity_size"):
            build_sds_targets(
                [np.zeros((2, 2), dtype=np.uint8)],
                num_phases=2,
                use_diffusivity=True,
            )

    def test_build_sds_targets_rejects_non_finite_numeric_options(self):
        image = np.zeros((2, 2), dtype=np.uint8)

        cases = [
            ("temperature", {"temperature": float("nan")}),
            ("sa_sigma", {"sa_sigma": float("nan"), "use_sa": True}),
            ("diffusivity_low_cond", {"diffusivity_low_cond": float("nan")}),
        ]

        for message, kwargs in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_sds_targets([image], num_phases=2, use_vf=True, **kwargs)

    def test_build_sds_targets_rejects_non_integer_size_options(self):
        image = np.zeros((2, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "sa_kernel_size.*integer"):
            build_sds_targets(
                [image],
                num_phases=2,
                use_sa=True,
                sa_kernel_size=True,
            )

        with self.assertRaisesRegex(ValueError, "diffusivity_size.*integer"):
            build_sds_targets(
                [image],
                num_phases=2,
                use_diffusivity=True,
                diffusivity_size=(2.5, 2),
            )


if __name__ == "__main__":
    unittest.main()
