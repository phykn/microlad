import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_device() -> tuple[torch.device, int, bool]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, False

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


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_distributed(
    model: torch.nn.Module,
    local_rank: int,
    distributed: bool,
) -> torch.nn.Module:
    if not distributed:
        return model

    if next(model.parameters()).device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[local_rank])

    return DistributedDataParallel(model)


