import torch


@torch.no_grad()
def multi_axis_decode(
    vae: torch.nn.Module,
    volume_z: torch.Tensor,
    downsample: int = 4,
) -> torch.Tensor:
    if volume_z.ndim != 4:
        raise ValueError("volume_z must have shape [C, D, H, W].")
    if downsample < 1:
        raise ValueError("downsample must be at least 1.")

    _, d, h, w = volume_z.shape
    sample = vae.decode(volume_z[:, 0, :, :].unsqueeze(0)).squeeze(0)
    out_c, full_h, full_w = sample.shape
    full_d = d * downsample
    decoded_acc = torch.zeros(
        (out_c, full_d, full_h, full_w), device=volume_z.device, dtype=sample.dtype
    )

    for zi in range(d):
        dec = vae.decode(volume_z[:, zi, :, :].unsqueeze(0)).squeeze(0)
        decoded_acc[:, zi * downsample : (zi + 1) * downsample, :, :] += dec.unsqueeze(
            1
        )

    for yi in range(h):
        dec = vae.decode(volume_z[:, :, yi, :].unsqueeze(0)).squeeze(0)
        decoded_acc[:, :, yi * downsample : (yi + 1) * downsample, :] += dec.unsqueeze(
            2
        )

    for xi in range(w):
        dec = vae.decode(volume_z[:, :, :, xi].unsqueeze(0)).squeeze(0)
        decoded_acc[:, :, :, xi * downsample : (xi + 1) * downsample] += dec.unsqueeze(
            3
        )

    return (decoded_acc / 3.0).permute(1, 0, 2, 3).clamp(0, 1)


@torch.no_grad()
def three_axis_refinement(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    refinement_steps: int,
) -> torch.Tensor:
    if volume.ndim != 4:
        raise ValueError("volume must have shape [D, C, H, W].")
    if volume.shape[1] != 1:
        raise ValueError("volume must have a single gray channel.")
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative.")

    result = volume
    d, _, h, w = result.shape
    for _ in range(refinement_steps):
        new_volume = torch.zeros_like(result)

        for zi in range(d):
            mu, _ = vae.encode(result[zi : zi + 1] * 2 - 1)
            dec = vae.decode(mu)[0, 0]
            new_volume[zi, 0] += dec

        for yi in range(h):
            plane = result[:, 0, yi, :].unsqueeze(0).unsqueeze(0)
            mu, _ = vae.encode(plane * 2 - 1)
            dec = vae.decode(mu)[0, 0]
            new_volume[:, 0, yi, :] += dec

        for xi in range(w):
            plane = result[:, 0, :, xi].unsqueeze(0).unsqueeze(0)
            mu, _ = vae.encode(plane * 2 - 1)
            dec = vae.decode(mu)[0, 0]
            new_volume[:, 0, :, xi] += dec

        result = (new_volume / 3.0).clamp(0, 1)

    return result
