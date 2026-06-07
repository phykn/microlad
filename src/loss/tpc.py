import torch
import torch.nn.functional as F


def setup_tpc_bins(
    height: int,
    width: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    radius = ((yy - height // 2) ** 2 + (xx - width // 2) ** 2).sqrt().round().long().view(-1)
    num_bins = int(radius.max().item()) + 1
    bin_mat = F.one_hot(radius, num_classes=num_bins).float().transpose(0, 1)
    bin_counts = bin_mat.sum(dim=1, keepdim=True)
    return bin_mat, bin_counts


def compute_tpc_torch(
    mask: torch.Tensor,
    bin_mat: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W].")
    height, width = mask.shape
    fft = torch.fft.fft2(mask)
    corr = torch.fft.ifft2(fft * torch.conj(fft)) / (height * width)
    corr = torch.real(torch.fft.fftshift(corr)).reshape(-1)
    return (bin_mat @ corr) / bin_counts.squeeze(1).clamp_min(1)


def compute_tpc_loss_ste(
    masks_p: torch.Tensor,
    phases: list[int],
    tpc_targets: dict[int, torch.Tensor],
    bin_mat: torch.Tensor,
    bin_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if masks_p.ndim != 3:
        raise ValueError("masks_p must have shape [P, H, W].")
    if not phases:
        raise ValueError("phases must not be empty.")

    loss = torch.tensor(0.0, device=device)
    hard_assignment = masks_p.argmax(dim=0)
    for phase_index, phase in enumerate(phases):
        hard_mask = (hard_assignment == phase_index).float()
        soft_mask = masks_p[phase_index]
        mask_ste = hard_mask - soft_mask.detach() + soft_mask
        pred = compute_tpc_torch(mask_ste, bin_mat, bin_counts)
        target = torch.as_tensor(tpc_targets[phase], device=device, dtype=pred.dtype)
        length = min(pred.shape[0], target.shape[0])
        loss = loss + F.mse_loss(pred[:length], target[:length])

    return loss / len(phases)


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


def build_grayscale_tpc_target(
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = _as_grayscale_image(condition).float()
    bin_mat, bin_counts = setup_tpc_bins(image.shape[-2], image.shape[-1], device=image.device)
    target = compute_tpc_torch(image, bin_mat, bin_counts).detach()
    return target, bin_mat, bin_counts


def build_grayscale_tpc_targets(
    conditions: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not conditions:
        raise ValueError("conditions must not be empty.")

    first_image = _as_grayscale_image(conditions[0]).float()
    bin_mat, bin_counts = setup_tpc_bins(
        first_image.shape[-2],
        first_image.shape[-1],
        device=first_image.device,
    )
    targets = [compute_tpc_torch(first_image, bin_mat, bin_counts).detach()]
    for condition in conditions[1:]:
        image = _as_grayscale_image(condition).float().to(first_image.device)
        if image.shape != first_image.shape:
            raise ValueError("all grayscale TPC conditions must have the same image shape.")
        targets.append(compute_tpc_torch(image, bin_mat, bin_counts).detach())
    return torch.stack(targets).mean(dim=0), bin_mat, bin_counts


def compute_grayscale_tpc_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    bin_mat: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    grayscale = _as_grayscale_image(image).float()
    pred = compute_tpc_torch(grayscale, bin_mat, bin_counts)
    target = target.to(device=pred.device, dtype=pred.dtype)
    length = min(pred.shape[0], target.shape[0])
    return F.mse_loss(pred[:length], target[:length])
