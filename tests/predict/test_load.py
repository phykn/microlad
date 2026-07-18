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
        self.assertTrue(hasattr(predictor.sampler.model, "anchor_encoder"))
        self.assertIn("axis_emb.weight", predictor.sampler.model.state_dict())
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

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            size=8,
            num_phases=2,
            base_ch=4,
            time_dim=8,
            timesteps=1,
            beta_start=0.01,
            beta_end=0.02,
        )


if __name__ == "__main__":
    unittest.main()
