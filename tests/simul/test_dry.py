import unittest

import numpy as np

from src.simul import make_dry_geometry, make_dry_volume


SMALL_SETTINGS = {
    "size": 32,
    "big_radius": 5,
    "big_fraction": 0.15,
    "small_fraction": 0.05,
}


def _assert_whole_analytic_primitives(
    test: unittest.TestCase,
    geometry,
) -> None:
    for index, particle in enumerate(geometry.particles):
        center = np.asarray(particle.center)
        axes = np.asarray(particle.axes)
        rotation = np.asarray(particle.rotation)
        bounds = np.ceil(np.abs(rotation) @ axes).astype(np.int32)
        z, y, x = np.meshgrid(
            np.arange(-bounds[0], bounds[0] + 1),
            np.arange(-bounds[1], bounds[1] + 1),
            np.arange(-bounds[2], bounds[2] + 1),
            indexing="ij",
        )
        offsets = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
        local = offsets @ rotation
        expected = np.sum((local / axes) ** 2, axis=1) <= 1.0 + 1e-12
        expected_offsets = offsets[expected]
        actual_coordinates = np.argwhere(geometry.instances == index)
        actual_offsets = actual_coordinates - center
        np.testing.assert_array_equal(actual_offsets, expected_offsets)

        test.assertTrue(np.all(actual_coordinates >= 0))
        test.assertTrue(
            np.all(actual_coordinates < geometry.labels.shape[0])
        )
        test.assertTrue(
            np.all(geometry.labels[tuple(actual_coordinates.T)] == particle.label)
        )


def _contact_pairs(instances: np.ndarray) -> set[tuple[int, int]]:
    pairs = set()
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left = instances[tuple(left_slice)]
        right = instances[tuple(right_slice)]
        touching = (left >= 0) & (right >= 0) & (left != right)
        for first, second in zip(left[touching], right[touching], strict=True):
            pairs.add(tuple(sorted((int(first), int(second)))))
    return pairs


class DryPackingTest(unittest.TestCase):
    def test_hard_spheres_are_whole_non_overlapping_and_may_touch(self):
        geometry = make_dry_geometry(
            **SMALL_SETTINGS,
            shape="sphere",
        )
        volume = geometry.labels
        instances = geometry.instances

        self.assertEqual(volume.dtype, np.uint8)
        self.assertEqual(instances.dtype, np.int32)
        self.assertTrue(np.array_equal(instances >= 0, volume > 0))
        self.assertTrue(np.isin(volume, (0, 1, 2)).all())
        self.assertEqual(
            len(np.unique(instances[instances >= 0])),
            len(geometry.particles),
        )
        _assert_whole_analytic_primitives(self, geometry)

        contacts = _contact_pairs(instances)
        self.assertEqual(len(contacts), geometry.report.particle_contacts)
        for particle in geometry.particles:
            np.testing.assert_allclose(
                particle.axes,
                np.full(3, particle.axes[0]),
            )
            np.testing.assert_array_equal(particle.rotation, np.eye(3))

        actual = np.bincount(volume.ravel(), minlength=3) / volume.size
        np.testing.assert_allclose(
            actual,
            geometry.report.achieved_fractions,
        )
        self.assertGreater(actual[1], 0.0)
        self.assertGreater(actual[2], 0.0)

    def test_each_call_draws_a_new_random_arrangement(self):
        settings = {
            "size": 20,
            "big_radius": 3,
            "big_fraction": 0.15,
            "small_fraction": 0.05,
            "shape": "sphere",
        }
        first_random = make_dry_volume(**settings)
        second_random = make_dry_volume(**settings)

        self.assertFalse(np.array_equal(first_random, second_random))

    def test_fraction_guides_may_stop_early_and_phases_stay_mixed(self):
        geometry = make_dry_geometry(
            size=48,
            big_radius=8,
            small_radius=3,
            big_fraction=0.30,
            small_fraction=0.45,
        )

        requested = geometry.report.requested_fractions
        achieved = geometry.report.achieved_fractions
        self.assertLess(achieved[1], requested[1])
        self.assertLess(achieved[2], requested[2])

        midpoint = geometry.labels.shape[0] // 2
        for label in (1, 2):
            centers = np.asarray(
                [
                    particle.center
                    for particle in geometry.particles
                    if particle.label == label
                ]
            )
            for axis in range(3):
                self.assertTrue(np.any(centers[:, axis] < midpoint))
                self.assertTrue(np.any(centers[:, axis] >= midpoint))

    def test_aligned_ellipsoids_are_whole_and_create_directional_bias(self):
        elongation = 2.2
        geometry = make_dry_geometry(
            **SMALL_SETTINGS,
            shape="aligned_ellipsoid",
            elongation=elongation,
            alignment_axis="z",
        )
        _assert_whole_analytic_primitives(self, geometry)

        for particle in geometry.particles:
            self.assertAlmostEqual(
                particle.axes[0] / particle.axes[1],
                elongation,
            )
            self.assertAlmostEqual(particle.axes[1], particle.axes[2])
            np.testing.assert_allclose(
                np.asarray(particle.rotation)[:, 0],
                np.asarray((1.0, 0.0, 0.0)),
                atol=1e-12,
            )

        occupied = geometry.labels > 0
        neighbor_rates = []
        for axis in range(3):
            left_slice = [slice(None)] * 3
            right_slice = [slice(None)] * 3
            left_slice[axis] = slice(None, -1)
            right_slice[axis] = slice(1, None)
            neighbor_rates.append(
                np.mean(
                    occupied[tuple(left_slice)]
                    & occupied[tuple(right_slice)]
                )
            )
        self.assertGreater(
            neighbor_rates[0],
            max(neighbor_rates[1:]) + 0.02,
        )

    def test_rejects_unknown_shape_and_axis(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            make_dry_volume(**SMALL_SETTINGS, shape="capsule")
        with self.assertRaisesRegex(ValueError, "alignment_axis"):
            make_dry_volume(
                **SMALL_SETTINGS,
                shape="aligned_ellipsoid",
                alignment_axis="diagonal",
            )


if __name__ == "__main__":
    unittest.main()
