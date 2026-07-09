import unittest

import torch

from src.scaling.blending import blend_window


class PredictBlendTest(unittest.TestCase):
    def test_blend_window_is_floored_finite_and_center_weighted(self):
        window = blend_window(
            5,
            4,
            device=torch.device("cpu"),
            dtype=torch.float64,
            floor=0.05,
        )

        self.assertEqual(window.shape, torch.Size([5, 4]))
        self.assertEqual(window.dtype, torch.float64)
        self.assertEqual(window.device.type, "cpu")
        self.assertTrue(torch.isfinite(window).all())
        self.assertGreaterEqual(float(window.min()), 0.05)
        self.assertGreater(float(window[2, 1]), float(window[0, 0]))

    def test_blend_window_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "height"):
            blend_window(0, 4, device=torch.device("cpu"), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "floor"):
            blend_window(4, 4, device=torch.device("cpu"), dtype=torch.float32, floor=0)


if __name__ == "__main__":
    unittest.main()
