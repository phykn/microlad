import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from src.build import build_model
from src.misc import save_config
from src.predict import MPDDOptions, MPDDPredictor, load_predictor


class PredictorLoadTest(unittest.TestCase):
    def test_loads_trained_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            args = self._args()
            source = build_model(args)
            checkpoint = run_dir / "weight" / "mpdd" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": source.state_dict()}, checkpoint)
            save_config(run_dir, args, name="model")

            predictor = load_predictor(run_dir, device="cpu")
            volume, _ = predictor.predict(
                MPDDOptions(
                    num_phases=2,
                    volume_size=8,
                    harmonization_steps=1,
                    progress=False,
                )
            )

        self.assertIsInstance(predictor, MPDDPredictor)
        self.assertEqual(volume.shape, torch.Size([8, 8, 8]))
        self.assertEqual(volume.dtype, torch.uint8)
        self.assertEqual(predictor.sampler.model.num_axis_conditions, 0)
        self.assertFalse(predictor.sampler.model.anchor_conditioning)
        self.assertEqual(predictor.sampler.model.anchor_release_step, 0)
        self.assertNotIn("axis_emb.weight", predictor.sampler.model.state_dict())
        self.assertFalse(predictor.sampler.model.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in predictor.sampler.model.parameters()
            )
        )

    def test_reports_missing_run_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "model config"):
                load_predictor(tmp, device="cpu")

    def test_reports_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            save_config(run_dir, self._args(), name="model")
            checkpoint = run_dir / "weight" / "mpdd" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"not a checkpoint")

            with self.assertRaisesRegex(ValueError, "could not be loaded"):
                load_predictor(run_dir, device="cpu")

    def test_loads_axis_conditioned_run_from_saved_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            args = self._args(num_axis_conditions=3)
            source = build_model(args)
            checkpoint = run_dir / "weight" / "mpdd" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": source.state_dict()}, checkpoint)
            save_config(run_dir, args, name="model")

            predictor = load_predictor(run_dir, device="cpu")
            volume, _ = predictor.predict(
                MPDDOptions(
                    num_phases=2,
                    volume_size=8,
                    harmonization_steps=1,
                    progress=False,
                )
            )

        self.assertEqual(predictor.sampler.model.num_axis_conditions, 3)
        self.assertIn("axis_emb.weight", predictor.sampler.model.state_dict())
        self.assertEqual(volume.shape, torch.Size([8, 8, 8]))

    def test_rejects_conditional_checkpoint_with_legacy_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = build_model(self._args(num_axis_conditions=3))
            checkpoint = run_dir / "weight" / "mpdd" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": source.state_dict()}, checkpoint)
            save_config(run_dir, self._args(), name="model")

            with self.assertRaisesRegex(ValueError, "could not be loaded"):
                load_predictor(run_dir, device="cpu")

    def test_rejects_legacy_checkpoint_with_conditional_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = build_model(self._args())
            checkpoint = run_dir / "weight" / "mpdd" / "last" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model": source.state_dict()}, checkpoint)
            save_config(
                run_dir,
                self._args(num_axis_conditions=3),
                name="model",
            )

            with self.assertRaisesRegex(ValueError, "could not be loaded"):
                load_predictor(run_dir, device="cpu")

    @staticmethod
    def _args(num_axis_conditions: int | None = None) -> argparse.Namespace:
        values = dict(
            size=8,
            num_phases=2,
            base_ch=4,
            time_dim=8,
            timesteps=1,
            beta_start=0.01,
            beta_end=0.02,
        )
        if num_axis_conditions is not None:
            values["num_axis_conditions"] = num_axis_conditions
        return argparse.Namespace(**values)


if __name__ == "__main__":
    unittest.main()
