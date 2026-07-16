import copy
from collections.abc import Iterable
from datetime import datetime
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..model import encode_labels
from ..misc import require_int
from .anchor import sample_anchor_condition
from .distributed import is_main, unwrap


class MPDDTrainer:
    def __init__(
        self,
        model: nn.Module,
        loader: Iterable,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        num_phases: int,
        steps: int,
        device: str | torch.device,
        run_root: str | Path = "run",
        run_dir: str | Path | None = None,
        save_every: int = 1,
        clip_grad_norm: float | None = 1.0,
        ema_decay: float = 0.999,
        condition_dropout: float = 0.1,
        anchor_empty_probability: float = 0.2,
        warmup_steps: int = 0,
    ) -> None:
        require_int("num_phases", num_phases)
        require_int("steps", steps)
        require_int("save_every", save_every)
        require_int("warmup_steps", warmup_steps)
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if save_every <= 0:
            raise ValueError("save_every must be positive.")
        if clip_grad_norm is not None and clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive or None.")
        if (
            not isinstance(ema_decay, (int, float))
            or isinstance(ema_decay, bool)
            or not math.isfinite(ema_decay)
            or not 0.0 <= ema_decay < 1.0
        ):
            raise ValueError(
                "ema_decay must be between zero inclusive and one exclusive."
            )
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between zero and one.")
        if (
            not isinstance(anchor_empty_probability, (int, float))
            or isinstance(anchor_empty_probability, bool)
            or not math.isfinite(anchor_empty_probability)
            or not 0.0 <= anchor_empty_probability <= 1.0
        ):
            raise ValueError(
                "anchor_empty_probability must be between zero and one."
            )
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")

        self.device = torch.device(device)
        self.num_phases = num_phases
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(unwrap(self.model)).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.loader = loader
        self.iterator = iter(loader)
        self.loss = loss
        self.optimizer = optimizer
        self.steps = steps
        self.save_every = save_every
        self.clip_grad_norm = clip_grad_norm
        self.ema_decay = float(ema_decay)
        self.condition_dropout = float(condition_dropout)
        self.anchor_empty_probability = float(anchor_empty_probability)
        self.warmup_steps = warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step = 0
        self.is_main = is_main()

        self._setup_run(run_root, run_dir)
        self._save()

    def train_step(self) -> dict[str, float]:
        self.model.train()
        image, fractions, axis_condition = self._next()
        clean = encode_labels(
            image.to(self.device),
            self.num_phases,
        )
        anchor_image = None
        anchor_mask = None
        if bool(getattr(unwrap(self.model), "anchor_conditioning", False)):
            anchor_image, anchor_mask = sample_anchor_condition(
                clean,
                empty_probability=self.anchor_empty_probability,
            )
        if axis_condition is not None:
            axis_condition = axis_condition.to(self.device)
        if fractions is not None:
            fractions = fractions.to(self.device)
            drop = (
                torch.rand(fractions.shape[0], device=self.device)
                < self.condition_dropout
            )
            fractions = fractions.clone()
            fractions[drop] = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        if anchor_image is not None and anchor_mask is not None:
            loss, parts = self.loss(
                self.model,
                clean,
                fractions=fractions,
                axis_condition=axis_condition,
                anchor_image=anchor_image,
                anchor_mask=anchor_mask,
            )
        elif fractions is None and axis_condition is None:
            loss, parts = self.loss(self.model, clean)
        elif axis_condition is None:
            loss, parts = self.loss(
                self.model,
                clean,
                fractions=fractions,
            )
        else:
            loss, parts = self.loss(
                self.model,
                clean,
                fractions=fractions,
                axis_condition=axis_condition,
            )
        loss.backward()
        grads = [
            param.grad for param in self.model.parameters() if param.grad is not None
        ]
        if self.clip_grad_norm is None:
            grad_norm = float(torch.nn.utils.get_total_norm(grads))
        else:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm,
                )
            )

        self._warmup()
        self.optimizer.step()
        self._update_ema()
        self.step += 1

        stats = {"loss": float(loss.detach().cpu())}
        stats.update(
            {name: float(value.detach().cpu()) for name, value in parts.items()}
        )
        stats["grad_norm"] = grad_norm
        if self.warmup_steps > 0:
            stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
        self._log(stats)
        self._save()
        return stats

    def train(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        progress = tqdm(
            range(self.steps),
            total=self.steps,
            desc="MPDD",
            disable=not self.is_main,
        )
        for _ in progress:
            stats = self.train_step()
            shown = {name: value for name, value in stats.items() if name != "noise"}
            progress.set_postfix(
                {name: f"{value:.4g}" for name, value in shown.items()}
            )
        return stats

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def _setup_run(
        self,
        run_root: str | Path,
        run_dir: str | Path | None,
    ) -> None:
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else Path(run_root) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        self.log_dir = self.run_dir / "log" / "mpdd"
        self.weight_dir = self.run_dir / "weight" / "mpdd"
        self.last_weight_dir = self.weight_dir / "last"
        self.writer: SummaryWriter | None = None
        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.last_weight_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _log(self, stats: dict[str, float]) -> None:
        if self.writer is None:
            return
        for name, value in stats.items():
            self.writer.add_scalar(f"train/{name}", value, self.step)

    def _save(self) -> None:
        if not self.is_main or self.step % self.save_every != 0:
            return
        ckpt = {
            "step": self.step,
            "model": unwrap(self.ema_model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        step_dir = self.weight_dir / str(self.step)
        _write(ckpt, step_dir / "model.pt")
        _write(ckpt, self.last_weight_dir / "model.pt")

    def _next(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration as exc:
                raise ValueError(
                    "loader is exhausted and cannot be restarted; "
                    "use a re-iterable loader or an infinite iterator."
                ) from exc

        if isinstance(batch, torch.Tensor):
            return batch, None, None
        if (
            isinstance(batch, (tuple, list))
            and batch
            and isinstance(batch[0], torch.Tensor)
        ):
            fractions = batch[1] if len(batch) > 1 else None
            if fractions is not None and not isinstance(
                fractions,
                torch.Tensor,
            ):
                raise TypeError("phase fractions must be a tensor.")
            axis_condition = batch[2] if len(batch) > 2 else None
            if axis_condition is not None and not isinstance(
                axis_condition,
                torch.Tensor,
            ):
                raise TypeError("axis condition must be a tensor.")
            return batch[0], fractions, axis_condition
        raise TypeError(
            "batch must be a tensor or a tuple/list whose first item is a tensor."
        )

    def _warmup(self) -> None:
        if self.warmup_steps <= 0:
            return
        scale = min((self.step + 1) / self.warmup_steps, 1.0)
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * scale

    @torch.no_grad()
    def _update_ema(self) -> None:
        online = unwrap(self.model)
        update = 1.0 - self.ema_decay
        for average, current in zip(
            self.ema_model.parameters(),
            online.parameters(),
            strict=True,
        ):
            average.lerp_(current.detach(), update)
        for average, current in zip(
            self.ema_model.buffers(),
            online.buffers(),
            strict=True,
        ):
            average.copy_(current.detach())


def _write(ckpt: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        torch.save(ckpt, temp)
        _replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _replace(source: Path, target: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.1)
