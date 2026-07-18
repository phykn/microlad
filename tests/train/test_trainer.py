import tempfile
import unittest
from unittest.mock import patch as mock_patch
from pathlib import Path

import torch

from src.diffusion import DDPMProcess, DiffusionLoss
from src.model import MPDDUNet
from src.train import MPDDTrainer
from src.train.distributed import unwrap


class TinyImageDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.seen_image = None
        self.seen_phase_fractions = None
        self.seen_axis_condition = None

    def forward(
        self,
        image,
        timestep,
        phase_fractions=None,
        axis_condition=None,
        *,
        anchor_image=None,
        anchor_mask=None,
    ):
        self.seen_image = image.detach().clone()
        self.seen_phase_fractions = phase_fractions
        self.seen_axis_condition = axis_condition
        return self.conv(image)


class WrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BigGradientDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(100.0))

    def forward(
        self,
        image,
        timestep,
        phase_fractions=None,
        axis_condition=None,
        *,
        anchor_image=None,
        anchor_mask=None,
    ):
        return torch.ones_like(image) * self.weight


class BigGradientLoss(torch.nn.Module):
    def forward(
        self,
        model,
        image,
        fractions=None,
        axis_condition=None,
        anchor_image=None,
        anchor_mask=None,
    ):
        timestep = torch.zeros(image.shape[0], dtype=torch.long)
        prediction = model(
            image,
            timestep,
            fractions,
            axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        loss = prediction.square().mean()
        return loss, {"noise": loss.detach()}


class RecordingLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module) -> None:
        super().__init__()
        self.loss = loss
        self.seen_image = None

    def forward(
        self,
        model,
        image,
        fractions=None,
        axis_condition=None,
        anchor_image=None,
        anchor_mask=None,
    ):
        self.seen_image = image.detach().clone()
        return self.loss(
            model,
            image,
            fractions=fractions,
            axis_condition=axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )


class MPDDTrainerTest(unittest.TestCase):
    def test_trains_centered_images_and_saves_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyImageDenoiser()
            trainer = self._make_trainer(model, tmp, warmup_steps=2)
            recording_loss = RecordingLoss(trainer.loss)
            trainer.loss = recording_loss

            stats = trainer.train_step()
            checkpoint = torch.load(
                trainer.weight_dir / "last" / "model.pt",
                weights_only=True,
            )
            trainer.close()

        self.assertEqual(trainer.weight_dir.name, "mpdd")
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(stats["lr"], 5e-4)
        self.assertEqual(
            set(torch.unique(recording_loss.seen_image).tolist()),
            {-1.0, 1.0},
        )

    def test_condition_dropout_uses_null_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyImageDenoiser()
            trainer = self._make_trainer(
                model,
                tmp,
                condition_dropout=1.0,
            )

            trainer.train_step()
            trainer.close()

        self.assertTrue(torch.equal(model.seen_phase_fractions, torch.zeros((2, 2))))

    def test_condition_dropout_preserves_axis_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyImageDenoiser()
            trainer = self._make_trainer(
                model,
                tmp,
                loader=[self._axis_batch()],
                condition_dropout=1.0,
            )

            trainer.train_step()
            trainer.close()

        self.assertTrue(torch.equal(model.seen_phase_fractions, torch.zeros((2, 2))))
        self.assertTrue(torch.equal(model.seen_axis_condition, torch.tensor([0, 2])))

    def test_axis_conditioned_mpdd_completes_real_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MPDDUNet(
                num_phases=2,
                image_size=8,
                base_ch=4,
                time_dim=8,
            )
            images = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float32)
            fractions = torch.tensor([[0.5, 0.5]]).expand(3, -1)
            axis_condition = torch.tensor([0, 1, 2])
            trainer = MPDDTrainer(
                model=model,
                loader=[(images, fractions, axis_condition)],
                loss=DiffusionLoss(
                    DDPMProcess(timesteps=2, beta_start=0.01, beta_end=0.02)
                ),
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                num_phases=2,
                steps=1,
                device="cpu",
                run_root=tmp,
            )

            stats = trainer.train_step()
            trainer.close()

        self.assertEqual(trainer.step, 1)
        self.assertTrue({"axis_0", "axis_1", "axis_2"}.issubset(stats))
        self.assertTrue(torch.all(model.axis_emb.weight.grad.abs().sum(dim=1) > 0))

    def test_checkpoint_uses_ema_and_unwrapped_parameter_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = WrappedModule(TinyImageDenoiser())
            trainer = self._make_trainer(model, tmp, ema_decay=0.5)
            initial = {
                name: value.detach().clone()
                for name, value in trainer.ema_model.state_dict().items()
            }

            trainer.train_step()
            checkpoint = torch.load(
                trainer.weight_dir / "last" / "model.pt",
                weights_only=True,
            )
            online = unwrap(trainer.model).state_dict()
            trainer.close()

        self.assertFalse(
            any(name.startswith("module.") for name in checkpoint["model"])
        )
        for name, value in checkpoint["model"].items():
            self.assertTrue(
                torch.allclose(value, initial[name].lerp(online[name], 0.5))
            )

    def test_restarts_reiterable_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyImageDenoiser()
            trainer = self._make_trainer(model, tmp, steps=2)

            stats = trainer.train()
            trainer.close()

        self.assertEqual(trainer.step, 2)
        self.assertIn("loss", stats)

    def test_rejects_exhausted_iterator(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyImageDenoiser()
            trainer = self._make_trainer(
                model,
                tmp,
                loader=iter([self._batch()]),
                steps=2,
            )

            with self.assertRaisesRegex(ValueError, "exhausted"):
                trainer.train()
            trainer.close()

    def test_gradient_clipping_reports_raw_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BigGradientDenoiser()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            trainer = MPDDTrainer(
                model=model,
                loader=[self._batch()],
                loss=BigGradientLoss(),
                optimizer=optimizer,
                num_phases=2,
                steps=1,
                device="cpu",
                run_root=tmp,
            )

            stats = trainer.train_step()
            clipped_norm = float(
                torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            trainer.close()

        self.assertGreater(stats["grad_norm"], 1.0)
        self.assertLessEqual(clipped_norm, 1.0001)

    def test_reuses_requested_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "existing"
            trainer = self._make_trainer(
                TinyImageDenoiser(),
                tmp,
                run_dir=run_dir,
            )

            self.assertEqual(trainer.run_dir, run_dir)
            self.assertTrue((run_dir / "log" / "mpdd").is_dir())
            trainer.close()

    def test_non_main_process_skips_run_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock_patch(
                "src.train.trainer.is_main",
                return_value=False,
            ):
                trainer = self._make_trainer(TinyImageDenoiser(), tmp)

            self.assertIsNone(trainer.writer)
            self.assertFalse(trainer.log_dir.exists())
            self.assertFalse(trainer.last_weight_dir.exists())
            trainer.close()

    def test_saves_checkpoints_at_requested_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(
                TinyImageDenoiser(),
                tmp,
                steps=2,
                save_every=2,
            )

            trainer.train_step()
            self.assertFalse((trainer.weight_dir / "1").exists())
            self.assertEqual(
                torch.load(
                    trainer.last_weight_dir / "model.pt",
                    weights_only=True,
                )["step"],
                0,
            )

            trainer.train_step()
            checkpoint = torch.load(
                trainer.last_weight_dir / "model.pt",
                weights_only=True,
            )
            self.assertTrue((trainer.weight_dir / "2" / "model.pt").is_file())
            trainer.close()

        self.assertEqual(checkpoint["step"], 2)

    def test_rejects_invalid_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, options in (
                ("num_phases", {"num_phases": 1}),
                ("steps", {"steps": 0}),
                ("save_every", {"save_every": 0}),
                ("ema_decay", {"ema_decay": 1.0}),
                ("condition_dropout", {"condition_dropout": 1.1}),
            ):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    self._make_trainer(TinyImageDenoiser(), tmp, **options)

    def _make_trainer(
        self,
        model: torch.nn.Module,
        run_root: str,
        *,
        loader=None,
        num_phases: int = 2,
        steps: int = 1,
        save_every: int = 1,
        ema_decay: float = 0.999,
        condition_dropout: float = 0.1,
        warmup_steps: int = 0,
        run_dir: Path | None = None,
    ) -> MPDDTrainer:
        return MPDDTrainer(
            model=model,
            loader=[self._batch()] if loader is None else loader,
            loss=DiffusionLoss(
                DDPMProcess(timesteps=2, beta_start=0.01, beta_end=0.02)
            ),
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            num_phases=num_phases,
            steps=steps,
            device="cpu",
            run_root=run_root,
            run_dir=run_dir,
            save_every=save_every,
            ema_decay=ema_decay,
            condition_dropout=condition_dropout,
            warmup_steps=warmup_steps,
        )

    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.tensor(
            [
                [[[0, 0], [1, 1]]],
                [[[1, 0], [1, 0]]],
            ],
            dtype=torch.float32,
        )
        fractions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        return images, fractions

    @classmethod
    def _axis_batch(
        cls,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, fractions = cls._batch()
        return images, fractions, torch.tensor([0, 2])


if __name__ == "__main__":
    unittest.main()
