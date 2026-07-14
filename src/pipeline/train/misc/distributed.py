import torch.distributed as dist
import torch.nn as nn


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
