import torch

from src.common.validation import require_int
from src.common.tensors.validation import validate_finite_tensor, validate_floating_dtype
from src.modeling.vae import get_downsample_factor


@torch.no_grad()
def generate_initial_volume(
    sampler,
    vae: torch.nn.Module,
    *,
    size: int | None = None,
    anchor_latent: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if size is None:
        size = int(vae.image_size)

    require_int("size", size)

    if size != int(vae.image_size):
        raise ValueError("size must match vae.image_size.")

    latent_ch = int(vae.latent_ch)
    latent_size = int(vae.latent_size)
    vae.eval()

    latent_batch = sampler.sample_lmpdd(
        (latent_size, latent_ch, latent_size, latent_size),
        anchor_latent=anchor_latent,
        anchor_mask=anchor_mask,
    )
    latent = latent_batch.permute(1, 0, 2, 3).contiguous()
    return decode_latent_volume(vae, latent)


def decode_latent(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latent)

    if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
        raise ValueError("vae.decode must return shape [1, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    validate_finite_tensor("decoded", decoded)

    return decoded[0, 0]


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents)

    if (
        decoded.ndim != 4
        or decoded.shape[0] != latents.shape[0]
        or decoded.shape[1] != 1
    ):
        raise ValueError("vae.decode must return shape [B, 1, H, W].")

    if decoded.shape[-2:] != (int(vae.image_size), int(vae.image_size)):
        raise ValueError("vae.decode output spatial shape must match vae.image_size.")

    validate_finite_tensor("decoded", decoded)

    return decoded[:, 0]


@torch.no_grad()
def decode_latent_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    _, depth, height, width = latent.shape
    factor = get_downsample_factor(vae)
    volume = torch.zeros(
        depth * factor,
        height * factor,
        width * factor,
        dtype=latent.dtype,
        device=latent.device,
    )

    for d in range(depth):
        decoded = decode_latent(vae, latent[:, d, :, :].unsqueeze(0)).float()
        volume[d * factor : (d + 1) * factor, :, :] += decoded.unsqueeze(0)

    for h in range(height):
        decoded = decode_latent(vae, latent[:, :, h, :].unsqueeze(0)).float()
        volume[:, h * factor : (h + 1) * factor, :] += decoded.unsqueeze(1)

    for w in range(width):
        decoded = decode_latent(vae, latent[:, :, :, w].unsqueeze(0)).float()
        volume[:, :, w * factor : (w + 1) * factor] += decoded.unsqueeze(2)

    return (volume / 3.0).float()


def _validate_latent_volume(vae: torch.nn.Module, latent: torch.Tensor) -> None:
    if latent.ndim != 4:
        raise ValueError("latent volume must have shape [C, D, H, W].")

    if latent.shape[0] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")

    validate_floating_dtype("latent dtype", latent.dtype)
    validate_finite_tensor("latent", latent)

    latent_size = int(vae.latent_size)

    if latent.shape[1:] != (latent_size, latent_size, latent_size):
        raise ValueError(
            f"latent spatial shape must be {(latent_size, latent_size, latent_size)}."
        )
