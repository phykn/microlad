import torch
import torch.nn.functional as F


def _as_grayscale_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        if image.shape[0] != 1 or image.shape[1] != 1:
            raise ValueError("batched grayscale image must have shape [1, 1, H, W].")
        return image[0, 0]
    if image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError("grayscale image must have shape [1, H, W].")
        return image[0]
    if image.ndim == 2:
        return image
    raise ValueError("image must have shape [H, W], [1, H, W], or [1, 1, H, W].")


def _as_tpc_target(
    target: torch.Tensor,
    prediction: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    target = torch.as_tensor(target, device=device, dtype=prediction.dtype)
    if target.shape != prediction.shape:
        raise ValueError("target length must match TPC prediction length.")
    return target


def build_tpc_bins(
    height: int,
    width: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if height <= 0:
        raise ValueError("height must be positive.")
    if width <= 0:
        raise ValueError("width must be positive.")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    radius = (
        ((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
        .sqrt()
        .round()
        .long()
        .view(-1)
    )
    num_bins = int(radius.max().item()) + 1
    bin_matrix = F.one_hot(radius, num_classes=num_bins).float().transpose(0, 1)
    bin_counts = bin_matrix.sum(dim=1, keepdim=True)
    return bin_matrix, bin_counts


def compute_tpc(
    mask: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    height, width = mask.shape
    fft = torch.fft.fft2(mask)
    corr = torch.fft.ifft2(fft * torch.conj(fft)) / (height * width)
    corr = torch.real(torch.fft.fftshift(corr)).reshape(-1)
    return (bin_matrix @ corr) / bin_counts.squeeze(1).clamp_min(1)


def build_grayscale_tpc_target(
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = _as_grayscale_image(condition).float()
    bin_matrix, bin_counts = build_tpc_bins(
        image.shape[-2], image.shape[-1], device=image.device
    )
    target = compute_tpc(image, bin_matrix, bin_counts).detach()
    return target, bin_matrix, bin_counts


def build_grayscale_tpc_targets(
    conditions: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not conditions:
        raise ValueError("conditions must not be empty.")

    first_image = _as_grayscale_image(conditions[0]).float()
    bin_matrix, bin_counts = build_tpc_bins(
        first_image.shape[-2],
        first_image.shape[-1],
        device=first_image.device,
    )
    targets = [compute_tpc(first_image, bin_matrix, bin_counts).detach()]
    for condition in conditions[1:]:
        image = _as_grayscale_image(condition).float().to(first_image.device)
        if image.shape != first_image.shape:
            raise ValueError(
                "all grayscale TPC conditions must have the same image shape."
            )
        targets.append(compute_tpc(image, bin_matrix, bin_counts).detach())
    return torch.stack(targets).mean(dim=0), bin_matrix, bin_counts


def compute_grayscale_tpc_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    grayscale = _as_grayscale_image(image).float()
    prediction = compute_tpc(grayscale, bin_matrix, bin_counts)
    target = _as_tpc_target(target, prediction, prediction.device)
    return F.mse_loss(prediction, target)
