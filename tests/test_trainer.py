import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from training.trainer import Trainer


class DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"x": torch.randn(1, 4, 4)}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


class DummyCriterion(nn.Module):
    def forward(self, model, batch):
        pred = model(batch["x"])
        loss = pred.mean().pow(2)
        return {"loss": loss}, loss


class EpochAwareSampler(Sampler):
    def __init__(self):
        self.epochs = []

    def __iter__(self):
        return iter([0, 1])

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class TrainerTest(unittest.TestCase):
    def make_trainer(self, tmp, accum_steps=1):
        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return Trainer(
            model=model,
            train_loader=DataLoader(DummyDataset(), batch_size=2),
            valid_loader=DataLoader(DummyDataset(), batch_size=2),
            criterion=DummyCriterion(),
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=tmp,
            max_grad_norm=1.0,
            accum_steps=accum_steps,
        )

    def test_step_returns_loss_dict_and_updates_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)
            before = trainer.model.conv.weight.detach().clone()

            losses = trainer.step()

            self.assertIn("loss", losses)
            self.assertIsInstance(losses["loss"], float)
            self.assertFalse(torch.allclose(before, trainer.model.conv.weight.detach()))

    def test_get_batch_wraps_around(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)

            for _ in range(4):
                batch = trainer.get_batch()

            self.assertIn("x", batch)

    def test_save_writes_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)

            trainer.save()

            self.assertTrue(os.path.exists(os.path.join(tmp, "weights", "last.pth")))

    def test_train_writes_tensorboard_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(tmp)

            trainer.train(steps=1, val_freq=1, save_freq=1)

            log_dir = os.path.join(tmp, "logs")
            self.assertTrue(os.path.exists(log_dir))
            self.assertTrue(any(name.startswith("events.out.tfevents") for name in os.listdir(log_dir)))

    def test_get_batch_sets_sampler_epoch_when_wrapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler = EpochAwareSampler()
            trainer = self.make_trainer(tmp)
            trainer.train_loader = DataLoader(DummyDataset(), batch_size=2, sampler=sampler)

            trainer.get_batch()
            trainer.get_batch()

            self.assertEqual(sampler.epochs, [0, 1])


if __name__ == "__main__":
    unittest.main()
