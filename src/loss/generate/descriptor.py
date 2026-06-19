import torch
import torch.nn.functional as F


def _validate_gray_image_batch(image: torch.Tensor) -> None:
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must be a gray image batch with shape [B, 1, H, W].")


def soft_gray_level_masks(
    image: torch.Tensor,
    levels: list[int],
    beta: float = 30.0,
) -> torch.Tensor:
    _validate_gray_image_batch(image)
    if not levels:
        raise ValueError("levels must not be empty.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    level_count = len(levels)
    level_values = torch.linspace(
        0.0, 1.0, level_count, device=image.device, dtype=image.dtype
    )
    batch, _, height, width = image.shape
    x = image.expand(batch, level_count, height, width)
    dist = torch.abs(x - level_values.view(1, level_count, 1, 1))
    return F.softmax(-beta * dist.view(batch, level_count, -1), dim=1).view(
        batch, level_count, height, width
    )


def compute_gray_mean(decoded: torch.Tensor) -> torch.Tensor:
    _validate_gray_image_batch(decoded)
    batch = decoded.shape[0]
    return decoded.view(batch, -1).mean(dim=1)


def compute_gray_moment_loss(
    decoded: torch.Tensor,
    target_mean: float,
    target_sqmean: float,
    mean_weight: float = 1.0,
    squared_mean_weight: float = 1.0,
) -> torch.Tensor:
    mean = compute_gray_mean(decoded)
    squared_mean = (decoded**2).view(decoded.shape[0], -1).mean(dim=1)

    mean_loss = F.mse_loss(mean, torch.full_like(mean, target_mean))
    squared_mean_loss = F.mse_loss(
        squared_mean, torch.full_like(squared_mean, target_sqmean)
    )
    return mean_weight * mean_loss + squared_mean_weight * squared_mean_loss


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
    gray_levels: list[int],
    kernel_size: int = 7,
    sigma: float = 1.0,
    beta: float = 50.0,
) -> torch.Tensor:
    _validate_gray_image_batch(recon)
    if not gray_levels:
        raise ValueError("gray_levels must not be empty.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    batch, _, height, width = recon.shape
    device = recon.device
    level_count = len(gray_levels)

    masks = soft_gray_level_masks(recon, gray_levels, beta=beta)

    kernel = _make_gaussian_kernel(kernel_size, sigma, device).repeat(
        level_count, 1, 1, 1
    )
    smooth = F.conv2d(
        masks, weight=kernel, padding=kernel_size // 2, groups=level_count
    )

    tv_h = torch.abs(smooth[:, :, 1:, :] - smooth[:, :, :-1, :]).sum(dim=(2, 3))
    tv_w = torch.abs(smooth[:, :, :, 1:] - smooth[:, :, :, :-1]).sum(dim=(2, 3))
    return ((tv_h + tv_w) / (height * width)).mean(dim=0)


def compute_surface_area_loss(
    decoded: torch.Tensor,
    surface_area_targets: dict[int, float],
    gray_levels: list[int],
    device: torch.device,
) -> torch.Tensor:
    relative_surface_area = compute_relative_surface_area(
        decoded, gray_levels, kernel_size=7, sigma=1.0
    )
    target = torch.tensor(
        [surface_area_targets[level] for level in gray_levels],
        device=device,
        dtype=relative_surface_area.dtype,
    )
    return F.mse_loss(relative_surface_area, target)


def compute_diffusivity_loss(
    masks: torch.Tensor,
    diffusivity_solver: torch.nn.Module,
    diffusivity_targets: dict[int, float],
    levels: list[int],
    device: torch.device,
) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError("masks must have shape [B, P, H, W].")
    if masks.shape[0] != 1:
        raise ValueError("compute_diffusivity_loss currently expects batch size 1.")
    if not levels:
        raise ValueError("levels must not be empty.")
    if masks.shape[1] < len(levels):
        raise ValueError("mask channels must cover every requested level.")

    solver_size = (
        getattr(diffusivity_solver, "height", masks.shape[-2]),
        getattr(diffusivity_solver, "width", masks.shape[-1]),
    )
    if masks.shape[-2:] != solver_size:
        masks = F.interpolate(
            masks, size=solver_size, mode="bilinear", align_corners=False
        )

    predicted_diffusivity = []
    for level_index in range(len(levels)):
        predicted_diffusivity.append(diffusivity_solver(masks[0, level_index]))
    prediction = torch.stack(predicted_diffusivity)
    target = torch.tensor(
        [diffusivity_targets[level] for level in levels],
        device=device,
        dtype=prediction.dtype,
    )
    return F.mse_loss(prediction, target)
