import math
from numbers import Real

import torch


def require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def require_finite_number(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def require_float(name: str, dtype: torch.dtype) -> None:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"{name} must be a floating point torch dtype.")


def require_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
