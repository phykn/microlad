import unittest

from src.scaling.tiles import tile_grid, tile_starts


class PredictScaleTilesTest(unittest.TestCase):
    def test_tile_starts_covers_tail_when_stride_does_not_divide_extent(self):
        self.assertEqual(
            tile_starts(5, tile_size=2, overlap=0),
            [0, 2, 3],
        )

    def test_tile_grid_uses_row_major_tile_order(self):
        self.assertEqual(
            list(tile_grid(4, 5, tile_size=2, overlap=0)),
            [(0, 0), (0, 2), (0, 3), (2, 0), (2, 2), (2, 3)],
        )

    def test_tile_starts_rejects_non_integer_inputs(self):
        cases = [
            (4.5, 2, 0),
            (4, "2", 0),
            (4, 2, True),
        ]

        for size, tile_size, overlap in cases:
            with self.subTest(size=size, tile_size=tile_size, overlap=overlap):
                with self.assertRaisesRegex(ValueError, "integer"):
                    tile_starts(size, tile_size=tile_size, overlap=overlap)

    def test_tile_starts_rejects_invalid_geometry(self):
        cases = [
            (0, 2, 0, "size"),
            (4, 0, 0, "tile_size"),
            (4, 2, -1, "overlap"),
            (4, 2, 2, "overlap"),
            (2, 4, 0, "fit"),
        ]

        for size, tile_size, overlap, message in cases:
            with self.subTest(size=size, tile_size=tile_size, overlap=overlap):
                with self.assertRaisesRegex(ValueError, message):
                    tile_starts(size, tile_size=tile_size, overlap=overlap)


if __name__ == "__main__":
    unittest.main()
