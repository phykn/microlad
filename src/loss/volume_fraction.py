import torch
import torch.nn.functional as F


def compute_volume_fraction(decoded: torch.Tensor) -> torch.Tensor:
    batch = decoded.shape[0]
    return decoded.view(batch, -1).mean(dim=1)


def compute_vf_loss(
    decoded: torch.Tensor,
    vf0: float,
    vf05: float,
    vf1: float,
    w_m1: float = 1.0,
    w_m2: float = 1.0,
) -> torch.Tensor:
    m1 = compute_volume_fraction(decoded)
    m2 = (decoded**2).view(decoded.shape[0], -1).mean(dim=1)

    target_mean = 0.5 * vf05 + vf1
    target_sqmean = (0.5**2) * vf05 + vf1

    loss_m1 = F.mse_loss(m1, torch.full_like(m1, target_mean))
    loss_m2 = F.mse_loss(m2, torch.full_like(m2, target_sqmean))

    return w_m1 * loss_m1 + w_m2 * loss_m2


def compute_vf_moment_loss(
    decoded: torch.Tensor,
    target_mean: float,
    target_sqmean: float,
    w_m1: float = 1.0,
    w_m2: float = 1.0,
) -> torch.Tensor:
    m1 = compute_volume_fraction(decoded)
    m2 = (decoded**2).view(decoded.shape[0], -1).mean(dim=1)

    loss_m1 = F.mse_loss(m1, torch.full_like(m1, target_mean))
    loss_m2 = F.mse_loss(m2, torch.full_like(m2, target_sqmean))
    return w_m1 * loss_m1 + w_m2 * loss_m2
