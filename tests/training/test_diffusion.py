import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.diffusion import DDPMProcess, DiffusionLoss
from src.training import DiffusionTrainer
from src.training.distributed import unwrap_model


def infinite_batches():
    while True:
        yield torch.randn(2, 1, 64, 64)


class TinyVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv2d(1, 4, kernel_size=4, stride=4)

    def encode(self, x: torch.Tensor):
        mu = self.encoder(x)
        return mu, torch.zeros_like(mu)


class TinyDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BigGradDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(100.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * self.weight


class BigGradLoss(torch.nn.Module):
    def forward(self, model, latent):
        t = torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device)
        pred = model(latent, t)
        loss = pred.pow(2).mean()
        return loss, {"noise": loss.detach()}


class FakeProgress:
    instances = []

    def __init__(self, iterable, **kwargs) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.postfixes = []
        self.__class__.instances.append(self)

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values) -> None:
        self.postfixes.append(values)


class DiffusionTrainerTest(unittest.TestCase):
    def test_train_step_updates_denoiser_and_keeps_vae_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            vae = TinyVAE()
            model = TinyDenoiser()
            loss_fn = DiffusionLoss(DDPMProcess(timesteps=4))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = DiffusionTrainer(
                model=model,
                vae=vae,
                dataloader=infinite_batches(),
                loss_fn=loss_fn,
                optimizer=optimizer,
                steps=1,
                device="cpu",
                run_root=tmp,
            )
            model_before = [parameter.detach().clone() for parameter in model.parameters()]
            vae_before = [parameter.detach().clone() for parameter in vae.parameters()]

            stats = trainer.train_step()
            trainer.close()

        model_changed = any(
            not torch.allclose(old, new.detach())
            for old, new in zip(model_before, model.parameters())
        )
        vae_changed = any(
            not torch.allclose(old, new.detach())
            for old, new in zip(vae_before, vae.parameters())
        )
        self.assertTrue(model_changed)
        self.assertFalse(vae_changed)
        self.assertTrue(all(not parameter.requires_grad for parameter in vae.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
        self.assertEqual(trainer.step, 1)
        self.assertIn("loss", stats)
        self.assertIn("noise", stats)
        self.assertIn("grad_norm", stats)
        self.assertGreaterEqual(stats["loss"], 0.0)
        self.assertGreater(stats["grad_norm"], 0.0)

    def test_trainer_can_reuse_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "existing-run"
            trainer = self._make_trainer(tmp, run_dir=run_dir)

            self.assertEqual(trainer.run_dir, run_dir)
            self.assertTrue((run_dir / "log" / "diffusion").is_dir())
            self.assertTrue((run_dir / "weight" / "diffusion" / "last").is_dir())
            trainer.close()

    def test_checkpoint_saves_unwrapped_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            vae = TinyVAE()
            model = WrappedModule(TinyDenoiser())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = DiffusionTrainer(
                model=model,
                vae=vae,
                dataloader=infinite_batches(),
                loss_fn=DiffusionLoss(DDPMProcess(timesteps=4)),
                optimizer=optimizer,
                steps=1,
                device="cpu",
                run_root=tmp,
            )

            trainer.train_step()
            trainer.close()
            checkpoint = torch.load(
                trainer.weight_dir / "last" / "model.pt",
                map_location="cpu",
            )

        self.assertIs(unwrap_model(model), model.module)
        self.assertFalse(any(key.startswith("module.") for key in checkpoint["model"]))

    def test_train_consumes_fixed_number_of_steps_from_infinite_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp, steps=2, save_every=2)

            stats = trainer.train()
            trainer.close()

            self.assertTrue((trainer.weight_dir / "2" / "model.pt").is_file())

        self.assertEqual(trainer.step, 2)
        self.assertIn("loss", stats)

    def test_train_restarts_finite_reiterable_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(
                tmp,
                dataloader=[torch.randn(2, 1, 64, 64)],
                steps=2,
                save_every=2,
            )

            stats = trainer.train()
            trainer.close()

        self.assertEqual(trainer.step, 2)
        self.assertIn("loss", stats)

    def test_train_shows_tqdm_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp, steps=2, save_every=2)
            FakeProgress.instances = []

            with patch("src.training.diffusion.tqdm", FakeProgress):
                stats = trainer.train()
            trainer.close()

        self.assertEqual(len(FakeProgress.instances), 1)
        progress = FakeProgress.instances[0]
        self.assertEqual(progress.kwargs["total"], 2)
        self.assertEqual(progress.kwargs["desc"], "diffusion")
        self.assertFalse(progress.kwargs["disable"])
        self.assertEqual(len(progress.postfixes), 2)
        self.assertEqual(set(progress.postfixes[-1]), {"loss", "grad_norm"})
        self.assertIn("loss", stats)

    def test_default_gradient_clipping_limits_gradients_after_logging_raw_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BigGradDenoiser()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            trainer = DiffusionTrainer(
                model=model,
                vae=TinyVAE(),
                dataloader=iter([torch.ones(2, 1, 64, 64)]),
                loss_fn=BigGradLoss(),
                optimizer=optimizer,
                steps=1,
                device="cpu",
                run_root=tmp,
            )

            stats = trainer.train_step()
            clipped_norm = trainer.grad_norm()
            trainer.close()

        self.assertGreater(stats["grad_norm"], 1.0)
        self.assertLessEqual(clipped_norm, 1.0001)

    def test_trainer_rejects_invalid_step_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "steps"):
                self._make_trainer(tmp, steps=0)
            with self.assertRaisesRegex(ValueError, "save_every"):
                self._make_trainer(tmp, save_every=0)

    def _make_trainer(
        self,
        run_root: str,
        steps: int = 1,
        save_every: int = 1,
        run_dir: Path | None = None,
        dataloader=None,
    ) -> DiffusionTrainer:
        model = TinyDenoiser()
        loss_fn = DiffusionLoss(DDPMProcess(timesteps=4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return DiffusionTrainer(
            model=model,
            vae=TinyVAE(),
            dataloader=infinite_batches() if dataloader is None else dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            steps=steps,
            device="cpu",
            run_root=run_root,
            run_dir=run_dir,
            save_every=save_every,
        )


if __name__ == "__main__":
    unittest.main()
