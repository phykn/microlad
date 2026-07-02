import torch

from src.predict.validation import validate_finite_tensor, validate_floating_dtype


@torch.no_grad()
def three_axis_refinement(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
) -> torch.Tensor:
    _validate_steps(steps)

    _validate_volume(volume, vae)

    refined = volume.clamp(-1.0, 1.0).float()

    if steps == 0:
        return refined

    vae.eval()

    for _ in range(steps):
        refined = _refine_once(refined, vae)

    return refined


def _refine_once(volume: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    depth, height, width = volume.shape
    new_volume = torch.zeros_like(volume)

    for z in range(depth):
        decoded = _encode_decode_slice(vae, volume[z].view(1, 1, height, width))
        new_volume[z, :, :] += decoded

    for y in range(height):
        decoded = _encode_decode_slice(vae, volume[:, y, :].view(1, 1, depth, width))
        new_volume[:, y, :] += decoded

    for x in range(width):
        decoded = _encode_decode_slice(vae, volume[:, :, x].view(1, 1, depth, height))
        new_volume[:, :, x] += decoded

    return (new_volume / 3.0).clamp(-1.0, 1.0).float()


def _encode_decode_slice(vae: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    mu, _ = vae.encode(image)

    if mu.ndim != 4:
        raise ValueError("encode output must have shape [B, C, H, W].")

    validate_finite_tensor("encoded latent", mu)

    decoded = vae.decode(mu)

    if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
        raise ValueError("decode output must have shape [1, 1, H, W].")

    if decoded.shape[-2:] != image.shape[-2:]:
        raise ValueError("decode output spatial shape must match the input slice.")

    validate_finite_tensor("decoded slice", decoded)

    return decoded[0, 0].float()


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
