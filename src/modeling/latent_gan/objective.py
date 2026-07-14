import torch


def critic_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    penalty: torch.Tensor,
    *,
    gp_weight: float = 10.0,
) -> torch.Tensor:
    """Computes the WGAN-GP critic objective."""
    if gp_weight < 0.0:
        raise ValueError("gp_weight must be non-negative.")
    if real_scores.numel() == 0 or fake_scores.numel() == 0:
        raise ValueError("critic scores must not be empty.")
    if penalty.ndim != 0:
        raise ValueError("penalty must be a scalar tensor.")

    return fake_scores.mean() - real_scores.mean() + gp_weight * penalty


def guidance_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Encourages decoded fake images to receive higher critic scores."""
    if fake_scores.numel() == 0:
        raise ValueError("critic scores must not be empty.")
    return -fake_scores.mean()


def gradient_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Penalizes critic gradients along real/fake interpolations."""
    if real.shape != fake.shape:
        raise ValueError("real and fake batches must have the same shape.")
    if real.ndim != 4 or real.shape[0] <= 0:
        raise ValueError("real and fake batches must have shape [B, C, H, W].")
    if not real.is_floating_point() or not fake.is_floating_point():
        raise ValueError("real and fake batches must be floating point.")
    if real.device != fake.device:
        raise ValueError("real and fake batches must be on the same device.")
    epsilon = torch.rand(
        real.shape[0],
        1,
        1,
        1,
        device=real.device,
        dtype=real.dtype,
    )
    mixed = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(
        outputs=critic(mixed).sum(),
        inputs=mixed,
        create_graph=True,
        only_inputs=True,
    )[0]
    norm = gradients.flatten(start_dim=1).norm(2, dim=1)
    return (norm - 1.0).square().mean()
