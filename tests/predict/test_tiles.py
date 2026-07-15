import unittest

import torch

from src.predict.tiles import iter_tiles, list_starts, make_window


class MPDDTilesTest(unittest.TestCase):
    def test_iter_tiles_covers_every_pixel(self):
        coverage = torch.zeros(9, 11, dtype=torch.int64)

        for row, col in iter_tiles(9, 11, tile_size=4, overlap=2):
            coverage[row : row + 4, col : col + 4] += 1

        self.assertTrue(torch.all(coverage > 0))

    def test_list_starts_covers_tail(self):
        self.assertEqual(list_starts(5, tile_size=2, overlap=0), [0, 2, 3])

    def test_make_window_is_finite_and_center_weighted(self):
        window = make_window(
            5,
            4,
            device=torch.device("cpu"),
            dtype=torch.float64,
            floor=0.05,
        )

        self.assertTrue(torch.isfinite(window).all())
        self.assertGreaterEqual(float(window.min()), 0.05)
        self.assertGreater(float(window[2, 1]), float(window[0, 0]))


if __name__ == "__main__":
    unittest.main()
