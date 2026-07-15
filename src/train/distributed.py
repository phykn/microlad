import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def setup() -> tuple[torch.device, int, bool]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return device, local_rank, True


def cleanup(enabled: bool) -> None:
    if enabled and is_active():
        dist.destroy_process_group()


def wrap(
    model: nn.Module,
    local_rank: int,
    enabled: bool,
) -> nn.Module:
    if not enabled:
        return model

    if next(model.parameters()).device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[local_rank])
    return DistributedDataParallel(model)


def is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_active():
        return 0
    return dist.get_rank()


def is_main() -> bool:
    return get_rank() == 0


def unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
