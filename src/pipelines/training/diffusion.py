from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.pipelines.training.misc.distributed import is_main_process, unwrap_model
from src.pipelines.training.misc.run import (
    log_stats,
    save_checkpoint,
    setup_run_dirs,
)


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        dataloader: Iterable,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int,
        device: str | torch.device,
        run_root: str | Path = "run",
        run_dir: str | Path | None = None,
        save_every: int = 1,
        clip_grad_norm: float | None = 1.0,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if save_every <= 0:
            raise ValueError("save_every must be positive.")
        if clip_grad_norm is not None and clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive or None.")

        self.model = model.to(device)
        self.vae = vae.to(device)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
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
            component="diffusion",
            is_main_process=self.is_main_process,
            run_dir=run_dir,
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=0,
            save_every=self.save_every,
            weight_dir=self.weight_dir,
            last_weight_dir=self.last_weight_dir,
            is_main_process=self.is_main_process,
        )

    def train_step(self) -> dict[str, float]:
        self.model.train()
        self.vae.eval()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            try:
                batch = next(self.iterator)
            except StopIteration as exc:
                raise ValueError(
                    "dataloader is exhausted and cannot be restarted; "
                    "use a re-iterable dataloader or an infinite iterator."
                ) from exc
        if isinstance(batch, torch.Tensor):
            image = batch
        elif isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            image = batch[0]
        else:
            raise TypeError(
                "batch must be a tensor or a tuple/list whose first item is a tensor."
            )
        image = image.to(self.device)

        with torch.no_grad():
            latent, _ = unwrap_model(self.vae).encode(image)

        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = self.loss_fn(self.model, latent)
        loss.backward()
        if self.clip_grad_norm is None:
            gradients = [
                parameter.grad
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            current_grad_norm = float(torch.nn.utils.get_total_norm(gradients))
        else:
            current_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm,
                )
            )

        self.optimizer.step()

        stats = {"loss": float(loss.detach().cpu())}
        stats.update(
            {name: float(value.detach().cpu()) for name, value in parts.items()}
        )
        stats["grad_norm"] = current_grad_norm
        self.step += 1
        log_stats(self.writer, stats, self.step)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            save_every=self.save_every,
            weight_dir=self.weight_dir,
            last_weight_dir=self.last_weight_dir,
            is_main_process=self.is_main_process,
        )
        return stats

    def train(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        progress = tqdm(
            range(self.steps),
            total=self.steps,
            desc="diffusion",
            disable=not self.is_main_process,
        )

        for _ in progress:
            stats = self.train_step()
            visible_stats = {name: value for name, value in stats.items() if name != "noise"}
            progress.set_postfix(
                {name: f"{value:.4g}" for name, value in visible_stats.items()}
            )

        return stats

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
