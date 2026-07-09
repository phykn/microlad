import tempfile
import unittest
from unittest.mock import patch

import torch

from src.loss import VAELoss
from src.models import PatchVAE
from src.train import VAETrainer
from src.train.distributed import unwrap_model


def infinite_batches():
    while True:
        yield torch.randint(0, 3, (2, 1, 64, 64), dtype=torch.float32)


class WrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BigGradModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(100.0))

    def forward(self, x: torch.Tensor):
        recon = x * self.weight
        mu = torch.zeros(1, 1, 1, 1, device=x.device)
        logvar = torch.zeros_like(mu)
        return recon, mu, logvar


class BigGradLoss(torch.nn.Module):
    def forward(self, recon, target, mu, logvar):
        loss = recon.pow(2).mean()
        parts = {
            "reconstruction": loss.detach(),
            "ssim": torch.zeros((), device=recon.device),
            "kl": torch.zeros((), device=recon.device),
        }
        return loss, parts


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


class VAETrainerTest(unittest.TestCase):
    def test_train_step_updates_model_and_returns_loss_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=16)
            loss_fn = VAELoss(beta=0.0, ssim_weight=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = VAETrainer(
                model=model,
                dataloader=infinite_batches(),
                loss_fn=loss_fn,
                optimizer=optimizer,
                steps=1,
                device="cpu",
                run_root=tmp,
            )
            before = [parameter.detach().clone() for parameter in model.parameters()]

            stats = trainer.train_step()
            trainer.close()

        changed = any(
            not torch.allclose(old, new.detach())
            for old, new in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertEqual(trainer.step, 1)
        self.assertEqual(
            set(stats.keys()),
            {"loss", "reconstruction", "ssim", "kl", "grad_norm"},
        )
        self.assertGreaterEqual(stats["loss"], 0.0)
        self.assertGreater(stats["grad_norm"], 0.0)

    def test_checkpoint_saves_unwrapped_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = WrappedModule(
                PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=16)
            )
            loss_fn = VAELoss(beta=0.0, ssim_weight=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = VAETrainer(
                model=model,
                dataloader=infinite_batches(),
                loss_fn=loss_fn,
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

    def test_train_step_writes_tensorboard_log_and_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp)

            trainer.train_step()
            trainer.close()
            checkpoint = torch.load(
                trainer.weight_dir / "last" / "model.pt",
                map_location="cpu",
            )

            event_files = list(
                (trainer.run_dir / "log" / "vae").glob("events.out.tfevents.*")
            )
            self.assertEqual(checkpoint["step"], 1)
            self.assertIn("model", checkpoint)
            self.assertIn("optimizer", checkpoint)

        self.assertTrue(event_files)

    def test_train_consumes_fixed_number_of_steps_from_infinite_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp, steps=2, save_every=2)

            stats = trainer.train()
            self.assertTrue((trainer.weight_dir / "2" / "model.pt").is_file())
            trainer.close()

        self.assertEqual(trainer.step, 2)
        self.assertIn("loss", stats)

    def test_train_restarts_finite_reiterable_dataloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(
                tmp,
                dataloader=[torch.randint(0, 3, (2, 1, 64, 64), dtype=torch.float32)],
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

            with patch("src.train.vae.tqdm", FakeProgress):
                stats = trainer.train()
            trainer.close()

        self.assertEqual(len(FakeProgress.instances), 1)
        progress = FakeProgress.instances[0]
        self.assertEqual(progress.kwargs["total"], 2)
        self.assertEqual(progress.kwargs["desc"], "vae")
        self.assertFalse(progress.kwargs["disable"])
        self.assertEqual(len(progress.postfixes), 2)
        self.assertEqual(
            set(progress.postfixes[-1]),
            {"loss", "reconstruction", "ssim", "kl", "grad_norm"},
        )
        self.assertIn("loss", stats)

    def test_checkpoint_interval_controls_step_checkpoint_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp, steps=3, save_every=2)

            trainer.train()
            last_checkpoint = torch.load(
                trainer.weight_dir / "last" / "model.pt",
                map_location="cpu",
            )
            trainer.close()

            self.assertFalse((trainer.weight_dir / "1" / "model.pt").exists())
            self.assertTrue((trainer.weight_dir / "2" / "model.pt").is_file())
            self.assertFalse((trainer.weight_dir / "3" / "model.pt").exists())
            self.assertEqual(last_checkpoint["step"], 3)

    def test_default_gradient_clipping_limits_gradients_after_logging_raw_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BigGradModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            trainer = VAETrainer(
                model=model,
                dataloader=iter([torch.ones(2, 1, 1, 1)]),
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

    def test_train_rejects_non_positive_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=16)
            loss_fn = VAELoss(beta=0.0, ssim_weight=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            with self.assertRaisesRegex(ValueError, "steps"):
                VAETrainer(
                    model=model,
                    dataloader=infinite_batches(),
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    steps=0,
                    device="cpu",
                    run_root=tmp,
                )

    def test_trainer_rejects_non_positive_save_every(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=16)
            loss_fn = VAELoss(beta=0.0, ssim_weight=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            with self.assertRaisesRegex(ValueError, "save_every"):
                VAETrainer(
                    model=model,
                    dataloader=infinite_batches(),
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    steps=1,
                    device="cpu",
                    run_root=tmp,
                    save_every=0,
                )

    def _make_trainer(
        self,
        run_root: str,
        steps: int = 1,
        save_every: int = 1,
        dataloader=None,
    ) -> VAETrainer:
        model = PatchVAE(image_size=64, latent_size=16, base_ch=8, max_ch=16)
        loss_fn = VAELoss(beta=0.0, ssim_weight=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return VAETrainer(
            model=model,
            dataloader=infinite_batches() if dataloader is None else dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            steps=steps,
            device="cpu",
            run_root=run_root,
            save_every=save_every,
        )


if __name__ == "__main__":
    unittest.main()
