import math
from numbers import Real


def require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def require_finite_number(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
