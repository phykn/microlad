from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.train.distributed import is_main_process
from src.train.utils import (
    image_from_batch,
    log_stats,
    loss_stats,
    model_grad_norm,
    progress_postfix,
    save_checkpoint,
    setup_run_dirs,
    validate_train_settings,
)


class VAETrainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int,
        device: str | torch.device,
        run_root: str | Path = "run",
        save_every: int = 1,
        clip_grad_norm: float | None = 1.0,
    ) -> None:
        validate_train_settings(
            steps=steps,
            save_every=save_every,
            clip_grad_norm=clip_grad_norm,
        )

        self.model = model.to(device)
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.steps = steps
        self.save_every = save_every
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device(device)
        self.step = 0
        self.is_main_process = is_main_process()

        (
            self.run_dir,
            self.log_dir,
            self.weight_dir,
            self.last_weight_dir,
            self.writer,
        ) = setup_run_dirs(
            run_root=run_root,
            component="vae",
            is_main_process=self.is_main_process,
        )

    def train_step(self) -> dict[str, float]:
        self.model.train()
        image = image_from_batch(next(self.iterator)).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = self.model(image)
        loss, parts = self.loss_fn(recon, image, mu, logvar)
        loss.backward()
        grad_norm = self.grad_norm()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        stats = loss_stats(loss, parts)
        stats["grad_norm"] = grad_norm
        self.step += 1
        self.log_step_stats(stats)
        self.save_checkpoint()
        return stats

    def train(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        progress = tqdm(
            range(self.steps),
            total=self.steps,
            desc="vae",
            disable=not self.is_main_process,
        )
        for _ in progress:
            stats = self.train_step()
            progress.set_postfix(progress_postfix(stats))
        return stats

    def log_step_stats(self, stats: dict[str, float]) -> None:
        log_stats(self.writer, stats, self.step)

    def save_checkpoint(self) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            save_every=self.save_every,
            weight_dir=self.weight_dir,
            last_weight_dir=self.last_weight_dir,
            is_main_process=self.is_main_process,
        )

    def grad_norm(self) -> float:
        return model_grad_norm(self.model.parameters())

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
