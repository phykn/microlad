import torch

from src.common.tensors.validation import validate_finite_tensor, validate_floating_dtype


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

    _validate_integer("size", size)

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


@torch.no_grad()
def decode_latent_volume(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    _validate_latent_volume(vae, latent)
    vae.eval()

    _, depth, height, width = latent.shape
    factor = _downsample_factor(vae)
    volume = torch.zeros(
        depth * factor,
        height * factor,
        width * factor,
        dtype=latent.dtype,
        device=latent.device,
    )

    for d in range(depth):
        decoded = _decode_slice(vae, latent[:, d, :, :].unsqueeze(0))
        volume[d * factor : (d + 1) * factor, :, :] += decoded.unsqueeze(0)

    for h in range(height):
        decoded = _decode_slice(vae, latent[:, :, h, :].unsqueeze(0))
        volume[:, h * factor : (h + 1) * factor, :] += decoded.unsqueeze(1)

    for w in range(width):
        decoded = _decode_slice(vae, latent[:, :, :, w].unsqueeze(0))
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


def _decode_slice(vae: torch.nn.Module, latent_slice: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latent_slice)

    if decoded.ndim != 4 or decoded.shape[:2] != (1, 1):
        raise ValueError("decode output must have shape [1, 1, H, W].")

    expected = (int(vae.image_size), int(vae.image_size))

    if tuple(decoded.shape[-2:]) != expected:
        raise ValueError(f"decode output spatial shape must be {expected}.")

    validate_finite_tensor("decoded", decoded)

    return decoded[0, 0].float()


def _downsample_factor(vae: torch.nn.Module) -> int:
    factor = int(
        getattr(
            vae,
            "downsample_factor",
            int(vae.image_size) // int(vae.latent_size),
        )
    )

    if factor <= 0:
        raise ValueError("VAE downsample factor must be positive.")

    if int(vae.image_size) != int(vae.latent_size) * factor:
        raise ValueError(
            "vae.image_size must equal vae.latent_size times downsample factor."
        )

    return factor


def _validate_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
