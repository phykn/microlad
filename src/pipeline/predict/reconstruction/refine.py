import torch

from src.modeling.phases.representation import geometric_probability_consensus
from src.validation import require_finite, require_float, require_int


@torch.no_grad()
def refine_probabilities(
    probabilities: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    batch_size: int = 16,
    strength: float = 1.0,
    anchor_strength: float | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    require_int("steps", steps)
    require_int("batch_size", batch_size)
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    num_phases = getattr(vae, "num_phases", None)
    if not isinstance(num_phases, int) or isinstance(num_phases, bool):
        raise ValueError("vae.num_phases must be an integer.")
    if num_phases < 2:
        raise ValueError("vae.num_phases must be at least 2.")
    if not callable(getattr(vae, "decode_probs", None)):
        raise ValueError("volume refinement requires vae.decode_probs.")

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")
    if anchor_strength is None:
        anchor_strength = strength
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError("anchor_strength must be between 0 and 1.")

    image_size = int(vae.image_size)
    expected = (1, num_phases, image_size, image_size, image_size)
    if probabilities.shape != expected:
        raise ValueError(f"probabilities must have shape {expected}.")
    require_float("probabilities dtype", probabilities.dtype)
    require_finite("probabilities", probabilities)
    if torch.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")
    normalizer = probabilities.sum(dim=1, keepdim=True)
    if torch.any(normalizer <= 0.0):
        raise ValueError("probabilities must have positive phase mass.")
    probabilities = probabilities.float() / normalizer
    if steps == 0:
        return probabilities

    if anchor_mask is not None:
        expected = (1, 1, *probabilities.shape[2:])
        if anchor_mask.shape != expected or anchor_mask.dtype != torch.bool:
            raise ValueError(f"anchor_mask must be boolean with shape {expected}.")
        anchor_mask = anchor_mask.to(device=probabilities.device)

    vae.eval()
    for _ in range(steps):
        refined = probabilities.argmax(dim=1)[0].float()
        projected = _refine_once(
            refined,
            vae,
            num_phases=num_phases,
            batch_size=batch_size,
        )
        blend = torch.full_like(probabilities[:, :1], strength)
        if anchor_mask is not None:
            blend = torch.where(
                anchor_mask,
                blend.new_full((), anchor_strength),
                blend,
            )
        tiny = torch.finfo(probabilities.dtype).tiny
        logits = (
            probabilities.clamp_min(tiny).log() * (1.0 - blend)
            + projected.clamp_min(tiny).log() * blend
        )
        probabilities = logits.softmax(dim=1)

    return probabilities


def _refine_once(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    num_phases: int,
    batch_size: int,
) -> torch.Tensor:
    depth, height, width = volume.shape

    depth_probs = _encode_decode_probs(
        vae,
        volume.reshape(depth, 1, height, width),
        num_phases=num_phases,
        batch_size=batch_size,
    ).permute(1, 0, 2, 3)
    height_probs = _encode_decode_probs(
        vae,
        volume.permute(1, 0, 2).contiguous().view(height, 1, depth, width),
        num_phases=num_phases,
        batch_size=batch_size,
    ).permute(1, 2, 0, 3)
    width_probs = _encode_decode_probs(
        vae,
        volume.permute(2, 0, 1).contiguous().view(width, 1, depth, height),
        num_phases=num_phases,
        batch_size=batch_size,
    ).permute(1, 2, 3, 0)

    return geometric_probability_consensus(
        torch.stack([depth_probs, height_probs, width_probs]),
        num_phases,
    ).unsqueeze(0)


def _encode_decode_probs(
    vae: torch.nn.Module,
    images: torch.Tensor,
    *,
    num_phases: int,
    batch_size: int,
) -> torch.Tensor:
    batches = []
    for image_batch in images.split(batch_size):
        mu, _ = vae.encode(image_batch)
        if mu.ndim != 4 or mu.shape[0] != image_batch.shape[0]:
            raise ValueError("encode output must have shape [B, C, H, W].")
        require_finite("encoded latent", mu)

        probabilities = vae.decode_probs(mu)
        expected_shape = (
            image_batch.shape[0],
            num_phases,
            *image_batch.shape[-2:],
        )
        if probabilities.shape != expected_shape:
            raise ValueError(
                "decode_probs output must have shape [B, num_phases, H, W]."
            )
        require_finite("decoded probabilities", probabilities)
        batches.append(probabilities.float())

    return torch.cat(batches)
