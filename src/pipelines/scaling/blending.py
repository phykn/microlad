import torch


def blend_window(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    floor: float = 1e-3,
) -> torch.Tensor:
    _require_positive_int("height", height)
    _require_positive_int("width", width)

    if floor <= 0.0:
        raise ValueError("floor must be positive.")

    window_h = torch.hann_window(
        height,
        periodic=False,
        device=device,
        dtype=dtype,
    )
    window_w = torch.hann_window(
        width,
        periodic=False,
        device=device,
        dtype=dtype,
    )

    return torch.outer(window_h, window_w).clamp_min(floor)


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")
