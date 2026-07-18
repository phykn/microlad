import unittest
from unittest.mock import patch

import numpy as np

from src.simul import make_geometry, make_volume


CFG = {
    "size": 32,
    "big_radius": 5,
    "big_fraction": 0.15,
    "small_fraction": 0.05,
}


def _assert_parts(test: unittest.TestCase, geo) -> None:
    for i, part in enumerate(geo.particles):
        ctr = np.asarray(part.center)
        axes = np.asarray(part.axes)
        bnd = np.ceil(axes).astype(np.int32)
        z, y, x = np.meshgrid(
            np.arange(-bnd[0], bnd[0] + 1),
            np.arange(-bnd[1], bnd[1] + 1),
            np.arange(-bnd[2], bnd[2] + 1),
            indexing="ij",
        )
        off = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
        keep = np.sum((off / axes) ** 2, axis=1) <= 1.0 + 1e-12
        want = off[keep] + ctr
        keep = np.all((want >= 0) & (want < geo.labels.shape[0]), axis=1)
        want = want[keep]
        pos = np.argwhere(geo.instances == i)
        np.testing.assert_array_equal(pos, want)

        test.assertTrue(np.all(pos >= 0))
        test.assertTrue(np.all(pos < geo.labels.shape[0]))
        test.assertTrue(
            np.all(geo.labels[tuple(pos.T)] == part.label)
        )


def _contacts(ids: np.ndarray) -> set[tuple[int, int]]:
    hits = set()
    for ax in range(3):
        ia = [slice(None)] * 3
        ib = [slice(None)] * 3
        ia[ax] = slice(None, -1)
        ib[ax] = slice(1, None)
        a = ids[tuple(ia)]
        b = ids[tuple(ib)]
        mask = (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[mask], b[mask], strict=True):
            hits.add(tuple(sorted((int(x), int(y)))))
    return hits


class GeometryTest(unittest.TestCase):
    def test_hard_spheres_are_cropped_non_overlapping_and_may_touch(self):
        geo = make_geometry(**CFG)
        vol = geo.labels
        ids = geo.instances

        self.assertEqual(vol.dtype, np.uint8)
        self.assertEqual(ids.dtype, np.int32)
        self.assertTrue(np.array_equal(ids >= 0, vol > 0))
        self.assertTrue(np.isin(vol, (0, 1, 2)).all())
        self.assertEqual(
            len(np.unique(ids[ids >= 0])),
            len(geo.particles),
        )
        _assert_parts(self, geo)

        hits = _contacts(ids)
        self.assertEqual(len(hits), geo.report.particle_contacts)
        for part in geo.particles:
            np.testing.assert_allclose(
                part.axes,
                np.full(3, part.axes[0]),
            )

        clipped = [
            p
            for p in geo.particles
            if np.any(np.asarray(p.center) - np.ceil(p.axes) < 0)
            or np.any(np.asarray(p.center) + np.ceil(p.axes) >= vol.shape[0])
        ]
        self.assertTrue(clipped)
        self.assertTrue(
            any(
                np.any(face)
                for ax in range(3)
                for face in (np.take(vol, 0, ax), np.take(vol, -1, ax))
            )
        )

        got = np.bincount(vol.ravel(), minlength=3) / vol.size
        np.testing.assert_allclose(
            got,
            geo.report.achieved_fractions,
        )
        self.assertGreater(got[1], 0.0)
        self.assertGreater(got[2], 0.0)

    def test_each_call_draws_a_new_random_arrangement(self):
        cfg = {
            "size": 20,
            "big_radius": 3,
            "big_fraction": 0.15,
            "small_fraction": 0.05,
        }
        a = make_volume(**cfg)
        b = make_volume(**cfg)

        self.assertFalse(np.array_equal(a, b))

    def test_fraction_guides_allow_local_variation_and_phases_stay_mixed(self):
        geo = make_geometry(
            size=48,
            big_radius=8,
            small_radius=3,
            big_fraction=0.30,
            small_fraction=0.45,
        )

        want = geo.report.requested_fractions
        got = geo.report.achieved_fractions
        self.assertGreater(got[1], 0.0)
        self.assertGreater(got[2], 0.0)
        self.assertNotEqual(got[1:], want[1:])
        self.assertLess(sum(got[1:]), 1.0)

        mid = geo.labels.shape[0] // 2
        for label in (1, 2):
            ctr = np.asarray(
                [
                    p.center
                    for p in geo.particles
                    if p.label == label
                ]
            )
            for ax in range(3):
                self.assertTrue(np.any(ctr[:, ax] < mid))
                self.assertTrue(np.any(ctr[:, ax] >= mid))

    def test_builds_192_then_crops_to_128(self):
        with patch("src.simul.geometry._place_phase") as place:
            geo = make_geometry(
                size=128,
                big_radius=20,
                small_radius=6,
                big_fraction=0.20,
                small_fraction=0.24,
            )

        self.assertEqual(place.call_args_list[0].kwargs["vol"].shape, (192,) * 3)
        self.assertEqual(geo.labels.shape, (128,) * 3)

    def test_only_big_particles_are_z_ellipses(self):
        elong = 2.2
        geo = make_geometry(
            **CFG,
            big_elongation=elong,
        )
        _assert_parts(self, geo)

        for part in geo.particles:
            if part.label == 1:
                np.testing.assert_allclose(
                    part.axes,
                    np.full(3, part.axes[0]),
                )
            else:
                self.assertAlmostEqual(
                    part.axes[0] / part.axes[1],
                    elong,
                )
                self.assertAlmostEqual(part.axes[1], part.axes[2])

        mask = geo.labels == 2
        rates = []
        for ax in range(3):
            ia = [slice(None)] * 3
            ib = [slice(None)] * 3
            ia[ax] = slice(None, -1)
            ib[ax] = slice(1, None)
            rates.append(np.mean(mask[tuple(ia)] & mask[tuple(ib)]))
        self.assertGreater(rates[0], max(rates[1:]) + 0.015)

    def test_rejects_invalid_big_elongation(self):
        with self.assertRaisesRegex(ValueError, "big_elongation"):
            make_volume(**CFG, big_elongation=0.9)


if __name__ == "__main__":
    unittest.main()
