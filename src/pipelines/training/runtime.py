from collections.abc import Iterable, Iterator
from datetime import datetime
import os
from pathlib import Path
import time
from uuid import uuid4

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.pipelines.training.distributed import unwrap_model


def validate_training(
    *,
    steps: int,
    save_every: int,
    clip_grad_norm: float | None,
) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive.")

    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    if clip_grad_norm is not None and clip_grad_norm <= 0:
        raise ValueError("clip_grad_norm must be positive or None.")


def setup_run_dirs(
    *,
    run_root: str | Path,
    component: str,
    is_main_process: bool,
    run_dir: str | Path | None = None,
) -> tuple[Path, Path, Path, Path, SummaryWriter | None]:
    root = Path(run_root)
    run_path = (
        Path(run_dir)
        if run_dir is not None
        else root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    log_dir = run_path / "log" / component
    weight_dir = run_path / "weight" / component
    last_weight_dir = weight_dir / "last"

    writer = None

    if is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        last_weight_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

    return run_path, log_dir, weight_dir, last_weight_dir, writer


def loss_stats(loss: torch.Tensor, parts: dict[str, torch.Tensor]) -> dict[str, float]:
    stats = {"loss": float(loss.detach().cpu())}
    stats.update({name: float(value.detach().cpu()) for name, value in parts.items()})
    return stats


def log_stats(writer: SummaryWriter | None, stats: dict[str, float], step: int) -> None:
    if writer is None:
        return

    for name, value in stats.items():
        writer.add_scalar(f"train/{name}", value, step)


def format_progress(stats: dict[str, float]) -> dict[str, str]:
    return {name: f"{value:.4g}" for name, value in stats.items()}


def next_batch(dataloader: Iterable, iterator: Iterator) -> tuple[object, Iterator]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise ValueError(
                "dataloader is exhausted and cannot be restarted; "
                "use a re-iterable dataloader or an infinite iterator."
            ) from exc


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_every: int,
    weight_dir: Path,
    last_weight_dir: Path,
    is_main_process: bool,
) -> None:
    if not is_main_process:
        return

    checkpoint = {
        "step": step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if step % save_every == 0:
        step_dir = weight_dir / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        write_checkpoint(checkpoint, step_dir / "model.pt")

    write_checkpoint(checkpoint, last_weight_dir / "model.pt")


def write_checkpoint(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temp_path)
        replace_atomic(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def replace_atomic(source: Path, target: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except OSError:
            if attempt == 4:
                raise

            time.sleep(0.1)


def calc_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue

        norm = parameter.grad.detach().data.norm(2).item()
        total += norm * norm

    return total**0.5


def unpack_batch(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch

    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]

    raise TypeError("batch must be a tensor or a tuple/list whose first item is a tensor.")


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
