import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        save_dir: str | Path,
        max_grad_norm: float = 1.0,
        accum_steps: int = 1,
        rank: int = 0,
    ) -> None:
        if accum_steps < 1:
            raise ValueError("accum_steps must be at least 1")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = str(save_dir)
        self.max_grad_norm = max_grad_norm
        self.accum_steps = accum_steps
        self.rank = rank
        self.device = next(model.parameters()).device

        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )
        self._writer: SummaryWriter | None = None
        self._train_iter = None
        self._sampler_epoch = 0

    def _set_sampler_epoch(self) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._sampler_epoch)
        self._sampler_epoch += 1

    def get_batch(self):
        if self._train_iter is None:
            self._set_sampler_epoch()
            self._train_iter = iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._set_sampler_epoch()
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accum: dict[str, float] = {}
        for _ in range(self.accum_steps):
            batch = self.get_batch()
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss, metrics = self.criterion(self.model, batch)
            self.scaler.scale(loss / self.accum_steps).backward()
            accum["loss"] = (
                accum.get("loss", 0.0) + float(loss.detach()) / self.accum_steps
            )
            for key, value in metrics.items():
                accum[key] = (
                    accum.get(key, 0.0) + float(value.detach()) / self.accum_steps
                )

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        return accum

    def _state_dict(self) -> dict:
        model = self.model.module if hasattr(self.model, "module") else self.model
        return {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }

    def save(self, name: str = "last.pth") -> None:
        if self.rank != 0:
            return
        weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self._state_dict(), os.path.join(weights_dir, name))

    def _write_scalars(self, split: str, step: int, metrics: dict[str, float]) -> None:
        if self.rank != 0 or self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(f"{split}/{key}", value, step)

    def train(self, steps: int, save_freq: int = 1000) -> None:
        if steps < 0:
            raise ValueError("steps must be at least 0")
        if save_freq < 1:
            raise ValueError("save_freq must be at least 1")

        if self.rank == 0:
            self._writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))

        try:
            for global_step in range(1, steps + 1):
                losses = self.step()
                self._write_scalars("train", global_step, losses)
                if self.rank == 0 and self._writer is not None:
                    self._writer.add_scalar(
                        "lr/lr", self.optimizer.param_groups[0]["lr"], global_step
                    )

                if global_step % save_freq == 0:
                    self.save()

            self.save()
        finally:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
                self._writer = None
