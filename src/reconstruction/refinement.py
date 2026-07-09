import torch

from src.tensors.validation import validate_finite_tensor, validate_floating_dtype


@torch.no_grad()
def three_axis_refinement(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
) -> torch.Tensor:
    _validate_steps(steps)

    _validate_volume(volume, vae)

    refined = volume.float()

    if steps == 0:
        return refined

    vae.eval()

    for _ in range(steps):
        refined = _refine_once(refined, vae)

    return refined


def _refine_once(volume: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    depth, height, width = volume.shape
    new_volume = torch.zeros_like(volume)

    decoded = _encode_decode_batch(
        vae,
        volume.reshape(depth, 1, height, width),
    )
    new_volume += decoded[:, 0, :, :]

    decoded = _encode_decode_batch(
        vae,
        volume.permute(1, 0, 2).contiguous().view(height, 1, depth, width),
    )
    new_volume += decoded[:, 0, :, :].permute(1, 0, 2)

    decoded = _encode_decode_batch(
        vae,
        volume.permute(2, 0, 1).contiguous().view(width, 1, depth, height),
    )
    new_volume += decoded[:, 0, :, :].permute(1, 2, 0)

    return (new_volume / 3.0).float()


def _encode_decode_batch(vae: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must have shape [B, 1, H, W].")

    mu, _ = vae.encode(images)

    if mu.ndim != 4:
        raise ValueError("encode output must have shape [B, C, H, W].")

    if mu.shape[0] != images.shape[0]:
        raise ValueError("encode output batch size must match the input batch.")

    validate_finite_tensor("encoded latent", mu)

    decoded = vae.decode(mu)

    if decoded.ndim != 4 or decoded.shape[:2] != (images.shape[0], 1):
        raise ValueError("decode output must have shape [B, 1, H, W].")

    if decoded.shape[-2:] != images.shape[-2:]:
        raise ValueError("decode output spatial shape must match the input slice.")

    validate_finite_tensor("decoded slice", decoded)

    return decoded.float()


def _validate_volume(volume: torch.Tensor, vae: torch.nn.Module) -> None:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")

    validate_floating_dtype("volume dtype", volume.dtype)
    validate_finite_tensor("volume", volume)

    depth, height, width = volume.shape

    if min(depth, height, width) <= 0:
        raise ValueError("volume dimensions must be positive.")

    if depth != height or depth != width:
        raise ValueError("three-axis refinement requires a cubic volume.")

    if depth != int(vae.image_size):
        raise ValueError("volume size must match vae.image_size.")


def _validate_steps(steps: int) -> None:
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")
