import numpy as np
import torch

from src.predict.types import MAX_UINT8_PHASES
from src.predict.validation import validate_finite_tensor, validate_floating_dtype


def quantize_phase(values: torch.Tensor, num_phases: int) -> torch.Tensor:
    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("num_phases must be an integer.")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 output."
        )

    validate_floating_dtype("values dtype", values.dtype)
    validate_finite_tensor("values", values)

    scaled = (values.clamp(-1.0, 1.0) + 1.0) * 0.5 * (num_phases - 1)
    return scaled.round().to(torch.uint8)


def model_output_to_phase(output: torch.Tensor, num_phases: int) -> torch.Tensor:
    if output.ndim != 4 or output.shape[1] != 1:
        raise ValueError("model output must have shape [B, 1, H, W].")

    return quantize_phase(output[:, 0], num_phases=num_phases)


def phase_to_numpy(phase: torch.Tensor) -> np.ndarray:
    if phase.dtype != torch.uint8:
        raise ValueError("phase must have dtype torch.uint8.")

    return phase.detach().cpu().numpy().astype(np.uint8, copy=False)
