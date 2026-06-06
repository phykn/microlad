import torch
import torch.nn.functional as F


def compute_vf_loss(
    decoded: torch.Tensor,
    vf0: float,
    vf05: float,
    vf1: float,
    w_m1: float,
    w_m2: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    batch = decoded.shape[0]
    m1 = decoded.view(batch, -1).mean(dim=1)
    m2 = (decoded**2).view(batch, -1).mean(dim=1)

    target_mean = 0.5 * vf05 + vf1
    target_sqmean = (0.5**2) * vf05 + vf1

    loss_m1 = F.mse_loss(m1, torch.full_like(m1, target_mean)) if w_m1 > 0 else torch.tensor(0.0, device=device)
    loss_m2 = F.mse_loss(m2, torch.full_like(m2, target_sqmean)) if w_m2 > 0 else torch.tensor(0.0, device=device)

    return loss_m1, loss_m2, float(m1.mean().detach())
