from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from src.modeling.critic import (
    LatentCritic,
    critic_loss,
    gradient_penalty,
)
from src.pipelines.guidance.critic.data import split_fake_bank, split_real_bank
from src.pipelines.guidance.critic.evaluate import (
    damage_slices,
    evaluate_critic,
    prepare_validation,
)

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
    if real_bank.ndim != 5:
        raise ValueError(
            "real_bank must have shape [sources, augmentations, channels, height, width]."
        )
    if fake_latents.ndim != 5 or fake_latents.shape[0] < 2:
        raise ValueError(
            "fake_latents must contain a holdout and at least one training volume."
        )
    if len(set(map(int, fake_latents.shape[-3:]))) != 1:
        raise ValueError("fake latent volumes must be cubic.")
    if real_bank.shape[2] != fake_latents.shape[1]:
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

    crop_size = int(real_bank.shape[-1])
    train_real, validation_real, source_holdout = split_real_bank(
        real_bank,
        validation_size=config.batch_size,
    )
    train_fake, validation_fake = split_fake_bank(
        fake_latents,
        crop_size=crop_size,
    )
    validation = prepare_validation(
        validation_real,
        validation_fake,
        batch_size=config.batch_size,
    )

    critic.train()
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
    )
    margins = []
    penalties = []
    losses = []
    best_state = None
    best_score = float("-inf")
    best_step = 0

    step_range = tqdm(
        range(config.steps),
        total=config.steps,
        desc="Latent critic",
        disable=not progress,
    )
    for step in step_range:
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
        damaged = damage_slices(real)
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
        completed = step + 1
        refresh_every = max(1, config.steps // 100)
        if progress and (
            completed == 1
            or completed % refresh_every == 0
            or completed == config.steps
        ):
            step_range.set_postfix(
                {
                    "loss": f"{float(loss.detach().item()):.4g}",
                    "margin": f"{float(margin.item()):.4g}",
                    "gp": f"{float(penalty.detach().item()):.4g}",
                }
            )

        should_validate = (
            completed % config.validate_every == 0 or completed == config.steps
        )
        if should_validate:
            metrics = evaluate_critic(critic, *validation)
            score = float(metrics["score"].item())
            if score > best_score:
                best_score = score
                best_step = completed
                best_state = {
                    name: value.detach().clone()
                    for name, value in critic.state_dict().items()
                }

    if best_state is None:
        best_state = {
            name: value.detach().clone()
            for name, value in critic.state_dict().items()
        }
        best_step = len(losses)
    critic.load_state_dict(best_state)
    metrics = evaluate_critic(critic, *validation)

    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    critic.eval()

    probe = validation[1].detach().requires_grad_(True)
    input_gradient = torch.autograd.grad(critic(probe).sum(), probe)[0]
    device = real_bank.device
    return {
        "critic_steps": torch.tensor(len(losses), device=device),
        "critic_best_step": torch.tensor(best_step, device=device),
        "critic_source_holdout": torch.tensor(source_holdout, device=device),
        "critic_loss": torch.stack(losses).mean(),
        "critic_margin": torch.stack(margins).mean(),
        "critic_gradient_penalty": torch.stack(penalties).mean(),
        "critic_validation_margin": metrics["validation_margin"],
        "critic_damage_margin": metrics["damage_margin"],
        "critic_shuffle_margin": metrics["shuffle_margin"],
        "critic_validation_accuracy": metrics["validation_accuracy"],
        "critic_damage_accuracy": metrics["damage_accuracy"],
        "critic_shuffle_accuracy": metrics["shuffle_accuracy"],
        "critic_stat_sensitivity": metrics["stat_sensitivity"],
        "critic_validation_score": metrics["score"],
        "critic_input_gradient_norm": input_gradient.flatten(1).norm(dim=1).mean(),
        "critic_input_gradient_finite": torch.isfinite(input_gradient).all(),
    }
