import numpy as np
import torch

from src.predict.types import MAX_UINT8_PHASES
from src.predict.validation import validate_finite_tensor, validate_floating_dtype


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

    validate_floating_dtype("values dtype", values.dtype)
    validate_finite_tensor("values", values)

    return values.clamp(0.0, float(num_phases - 1)).round().to(torch.uint8)


def model_output_to_phase(output: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if output.ndim != 4:
        raise ValueError(
            "model output must have shape [B, 1, H, W] or [B, num_phases, H, W]."
        )

    validate_floating_dtype("model output dtype", output.dtype)
    validate_finite_tensor("model output", output)

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
