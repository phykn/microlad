import unittest

import torch

from src.pipelines.guidance.sds.schedule import (
    build_balanced_schedule,
    select_slice,
    select_slice_batch,
)


class ScheduleTest(unittest.TestCase):
    def test_balanced_schedule_visits_every_slice_once_per_sweep(self):
        schedule = build_balanced_schedule(
            steps=6,
            batch_size=1,
            volume_size=2,
        )

        self.assertEqual(len(schedule), 6)
        self.assertEqual(
            set(schedule),
            {(axis, index) for axis in range(3) for index in range(2)},
        )

    def test_select_slice_rejects_invalid_step_and_schedule_entries(self):
        volume = torch.zeros(3, 3, 3)

        for step in (-1, 1.5):
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "step"):
                    select_slice(volume, step=step, schedule=None)

        with self.assertRaisesRegex(ValueError, "entry for each step"):
            select_slice(volume, step=1, schedule=[(0, 0)])
        with self.assertRaisesRegex(ValueError, "axis.*integer"):
            select_slice(volume, step=0, schedule=[(0.0, 1)])
        with self.assertRaisesRegex(ValueError, "index.*integer"):
            select_slice(volume, step=0, schedule=[(0, "1")])

    def test_select_slice_batch_rejects_invalid_schedule(self):
        volume = torch.zeros(3, 3, 3)

        with self.assertRaisesRegex(ValueError, "same axis"):
            select_slice_batch(
                volume,
                step=0,
                schedule=[(0, 0), (1, 1)],
                batch_size=2,
            )
        with self.assertRaisesRegex(ValueError, "axis.*integer"):
            select_slice_batch(
                volume,
                step=0,
                schedule=[(0, 0), (1.0, 1)],
                batch_size=2,
            )
        with self.assertRaisesRegex(ValueError, "index.*integer"):
            select_slice_batch(
                volume,
                step=0,
                schedule=[(0, 0), (0, True)],
                batch_size=2,
            )

    def test_select_slice_batch_uses_axes_large_enough_for_batch(self):
        axis, indices = select_slice_batch(
            torch.zeros(1, 3, 1),
            step=0,
            schedule=None,
            batch_size=2,
        )

        self.assertEqual(axis, 1)
        self.assertEqual(len(indices), 2)
