import torch
import torch.nn.functional as F


def _make_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    axis = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def compute_relative_surface_area(
    recon: torch.Tensor,
    phases: list[int],
    kernel_size: int = 7,
    sigma: float = 1.0,
    beta: float = 50.0,
) -> torch.Tensor:
    batch, _, height, width = recon.shape
    device, dtype = recon.device, recon.dtype
    phase_count = len(phases)

    levels = torch.linspace(0.0, 1.0, steps=phase_count, device=device, dtype=dtype)
    x = recon.expand(batch, phase_count, height, width)
    dist = torch.abs(x - levels.view(1, phase_count, 1, 1))
    masks = F.softmax(-beta * dist.view(batch, phase_count, -1), dim=1).view(batch, phase_count, height, width)

    kernel = _make_gaussian_kernel(kernel_size, sigma, device).repeat(phase_count, 1, 1, 1)
    smooth = F.conv2d(masks, weight=kernel, padding=kernel_size // 2, groups=phase_count)

    tv_h = torch.abs(smooth[:, :, 1:, :] - smooth[:, :, :-1, :]).sum(dim=(2, 3))
    tv_w = torch.abs(smooth[:, :, :, 1:] - smooth[:, :, :, :-1]).sum(dim=(2, 3))
    return ((tv_h + tv_w) / (height * width)).mean(dim=0)


def compute_sa_loss(
    decoded: torch.Tensor,
    sa_targets: dict[int, float],
    phases: list[int],
    device: torch.device,
) -> torch.Tensor:
    rel_sa = compute_relative_surface_area(decoded, phases, kernel_size=7, sigma=1.0)
    target = torch.tensor([sa_targets[phase] for phase in phases], device=device, dtype=rel_sa.dtype)
    return F.mse_loss(rel_sa, target)
