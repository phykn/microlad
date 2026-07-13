from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.modeling.phases.representation import phase_levels


_AXIS_ORDERS = (
    (0, 1, 2),
    (2, 1, 0),
    (1, 0, 2),
    (2, 0, 1),
    (0, 2, 1),
    (1, 2, 0),
)


def select_slices(
    step: int,
    *,
    size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[int, list[int]]:
    cycle = step // 3
    axis = _AXIS_ORDERS[cycle % len(_AXIS_ORDERS)][step % 3]
    indices = torch.randperm(size, device=device)[:batch_size].tolist()
    return axis, [int(index) for index in indices]


def extract_slices(
    probabilities: torch.Tensor,
    *,
    axis: int,
    indices: Sequence[int],
) -> torch.Tensor:
    index = torch.as_tensor(indices, device=probabilities.device, dtype=torch.long)
    if axis == 0:
        return probabilities[:, index, :, :].permute(1, 0, 2, 3)
    if axis == 1:
        return probabilities[:, :, index, :].permute(2, 0, 1, 3)
    return probabilities[:, :, :, index].permute(3, 0, 1, 2)


def phase_values(probabilities: torch.Tensor, *, num_phases: int) -> torch.Tensor:
    categorical = straight_through_one_hot(probabilities)
    levels = phase_levels(
        num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return (categorical * levels.view(1, num_phases, 1, 1)).sum(
        dim=1,
        keepdim=True,
    )


def straight_through_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    indices = probabilities.argmax(dim=1)
    hard = F.one_hot(indices, num_classes=int(probabilities.shape[1])).movedim(-1, 1)
    return hard.to(probabilities.dtype) + probabilities - probabilities.detach()
