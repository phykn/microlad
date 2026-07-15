import torch
import torch.nn.functional as F


MAX_PHASES = 256


def encode_labels(labels: torch.Tensor, num_phases: int) -> torch.Tensor:
    """Converts categorical images or volumes to channels in ``[-1, 1]``."""
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if labels.ndim < 3 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B, 1, ...spatial].")
    if any(size <= 0 for size in labels.shape):
        raise ValueError("labels dimensions must be positive.")
    if not torch.isfinite(labels).all():
        raise ValueError("labels must be finite.")
    if not torch.equal(labels, labels.round()):
        raise ValueError("labels must contain integer phase values.")
    if labels.min().item() < 0 or labels.max().item() >= num_phases:
        raise ValueError(f"labels must contain values from 0 to {num_phases - 1}.")

    encoded = F.one_hot(
        labels[:, 0].to(torch.long),
        num_classes=num_phases,
    ).movedim(-1, 1)
    return encoded.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
