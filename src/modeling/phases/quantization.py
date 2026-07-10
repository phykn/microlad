import numpy as np
import torch

from src.common.tensors.validation import require_finite, require_float


MAX_UINT8_PHASES = int(np.iinfo(np.uint8).max) + 1


def _validate_num_phases(num_phases: int) -> None:
    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("num_phases must be an integer.")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 output."
        )


def quantize_phase(values: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_num_phases(num_phases)

    require_float("values dtype", values.dtype)
    require_finite("values", values)

    return values.clamp(0.0, float(num_phases - 1)).round().to(torch.uint8)


def decode_phase(output: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if output.ndim != 4:
        raise ValueError(
            "model output must have shape [B, 1, H, W] or [B, num_phases, H, W]."
        )

    require_float("model output dtype", output.dtype)
    require_finite("model output", output)

    if output.shape[1] == 1:
        return quantize_phase(output[:, 0], num_phases=num_phases)

    if output.shape[1] == num_phases:
        return output.argmax(dim=1).to(torch.uint8)

    raise ValueError(
        "model output must have shape [B, 1, H, W] or [B, num_phases, H, W]."
    )


def phase_to_numpy(phase: torch.Tensor) -> np.ndarray:
    if phase.dtype != torch.uint8:
        raise ValueError("phase must have dtype torch.uint8.")

    return phase.detach().cpu().numpy().astype(np.uint8, copy=False)
