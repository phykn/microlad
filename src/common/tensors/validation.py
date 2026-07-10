import torch


def require_float(name: str, dtype: torch.dtype) -> None:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"{name} must be a floating point torch dtype.")


def require_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
