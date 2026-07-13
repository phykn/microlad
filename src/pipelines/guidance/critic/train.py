from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from src.modeling.critic import (
    LatentCritic,
    critic_loss,
    gradient_penalty,
)
from src.pipelines.guidance.critic.data import slice_latents

if TYPE_CHECKING:
    from src.app.api.options import CriticConfig


def train_critic(
    critic: LatentCritic,
    real_bank: torch.Tensor,
    fake_latents: torch.Tensor,
    *,
    config: "CriticConfig",
    progress: bool = False,
) -> dict[str, torch.Tensor]:
    if real_bank.ndim != 4:
        raise ValueError("real_bank must have shape [N, C, H, W].")
    if fake_latents.ndim != 5:
        raise ValueError("fake_latents must have shape [N, C, D, H, W].")
    if len(set(map(int, fake_latents.shape[-3:]))) != 1:
        raise ValueError("fake latent volumes must be cubic.")
    if real_bank.shape[1] != fake_latents.shape[1]:
        raise ValueError("real and fake latent channels must match.")
    if config.steps <= 0:
        raise ValueError("critic training steps must be positive.")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean.")
    if real_bank.device != fake_latents.device:
        raise ValueError("real and fake latents must be on the same device.")
    if real_bank.dtype != fake_latents.dtype:
        raise ValueError("real and fake latents must have the same dtype.")
    if not torch.isfinite(real_bank).all() or not torch.isfinite(fake_latents).all():
        raise ValueError("real and fake latents must be finite.")

    critic.train()
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
    )
    margins = []
    penalties = []
    losses = []
    crop_size = int(real_bank.shape[-1])
    train_real, validation_real = _split_bank(real_bank, config.batch_size)
    fake_bank = slice_latents(fake_latents, crop_size=crop_size)
    train_fake, validation_fake = _split_bank(fake_bank, config.batch_size)
    steps = tqdm(
        range(config.steps),
        total=config.steps,
        desc="Latent critic",
        disable=not progress,
    )
    for _ in steps:
        real_indices = torch.randint(
            0,
            train_real.shape[0],
            (config.batch_size,),
            device=real_bank.device,
        )
        fake_indices = torch.randint(
            0,
            train_fake.shape[0],
            (config.batch_size,),
            device=real_bank.device,
        )
        real = train_real[real_indices]
        generated = train_fake[fake_indices]
        damaged = _damage_slices(real)
        damaged_count = config.batch_size // 2
        fake = torch.cat(
            [generated[: config.batch_size - damaged_count], damaged[:damaged_count]]
        )

        optimizer.zero_grad(set_to_none=True)
        real_scores = critic(real)
        fake_scores = critic(fake)
        penalty = gradient_penalty(critic, real, fake)
        if (
            not torch.isfinite(real_scores).all()
            or not torch.isfinite(fake_scores).all()
            or not torch.isfinite(penalty)
        ):
            raise RuntimeError("critic training produced non-finite values.")
        loss = critic_loss(
            real_scores,
            fake_scores,
            penalty,
            gradient_weight=config.gradient_weight,
        )
        loss.backward()
        optimizer.step()

        margin = (real_scores.mean() - fake_scores.mean()).detach()
        margins.append(margin)
        penalties.append(penalty.detach())
        losses.append(loss.detach())
        completed = len(losses)
        refresh_every = max(1, config.steps // 100)
        if progress and (
            completed == 1
            or completed % refresh_every == 0
            or completed == config.steps
        ):
            steps.set_postfix(
                {
                    "loss": f"{float(loss.detach().item()):.4g}",
                    "margin": f"{float(margin.item()):.4g}",
                }
            )

    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    critic.eval()

    with torch.no_grad():
        count = min(
            config.batch_size,
            int(validation_real.shape[0]),
            int(validation_fake.shape[0]),
        )
        real = validation_real[-count:]
        fake = validation_fake[-count:]
        validation_margin = critic(real).mean() - critic(fake).mean()
        damage_margin = critic(real).mean() - critic(_damage_slices(real)).mean()

    probe = fake.detach().requires_grad_(True)
    input_gradient = torch.autograd.grad(critic(probe).sum(), probe)[0]
    gradient_finite = torch.isfinite(input_gradient).all()

    return {
        "critic_steps": torch.tensor(config.steps, device=real_bank.device),
        "critic_loss": torch.stack(losses).mean(),
        "critic_margin": torch.stack(margins).mean(),
        "critic_gradient_penalty": torch.stack(penalties).mean(),
        "critic_validation_margin": validation_margin,
        "critic_damage_margin": damage_margin,
        "critic_input_gradient_norm": input_gradient.flatten(1).norm(dim=1).mean(),
        "critic_input_gradient_finite": gradient_finite,
    }


def _damage_slices(latents: torch.Tensor) -> torch.Tensor:
    damaged = latents.clone()
    shift = max(1, int(latents.shape[-1]) // 2)
    damaged[:, :, 1::2] = torch.roll(
        damaged[:, :, 1::2],
        shifts=shift,
        dims=-1,
    )
    return damaged


def _split_bank(
    bank: torch.Tensor,
    validation_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bank.shape[0] < 2:
        raise ValueError("critic banks must contain at least two samples.")
    order = torch.randperm(bank.shape[0], device=bank.device)
    bank = bank[order]
    validation_size = min(
        max(1, validation_size),
        max(1, int(bank.shape[0]) // 4),
        int(bank.shape[0]) - 1,
    )
    return bank[:-validation_size], bank[-validation_size:]
