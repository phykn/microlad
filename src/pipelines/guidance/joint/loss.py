import torch
import torch.nn.functional as F

def anchor_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask > 0
    selected = probabilities.permute(1, 2, 3, 0)[active]
    indices = target[active].round().to(torch.long)
    return F.nll_loss(
        selected.clamp_min(torch.finfo(selected.dtype).tiny).log(),
        indices,
    )


def continuity_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if min(probabilities.shape[2:]) < 3:
        return probabilities.sum() * 0.0
    smoothed = F.avg_pool3d(
        probabilities,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )
    curvature = []
    for dimension in (2, 3, 4):
        length = int(smoothed.shape[dimension])
        for lag in (1, 2, 3):
            span = length - 2 * lag
            if span <= 0:
                continue
            before = smoothed.narrow(dimension, 0, span)
            middle = smoothed.narrow(dimension, lag, span)
            after = smoothed.narrow(dimension, 2 * lag, span)
            curvature.append((after - 2.0 * middle + before).abs().mean())
    return torch.stack(curvature).mean()
