import torch

from src.modeling.critic import LatentCritic


def prepare_validation(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    count = min(batch_size, int(real.shape[0]), int(fake.shape[0]))
    real_indices = torch.linspace(
        0,
        int(real.shape[0]) - 1,
        steps=count,
        device=real.device,
    ).round().long()
    fake_indices = torch.linspace(
        0,
        int(fake.shape[0]) - 1,
        steps=count,
        device=fake.device,
    ).round().long()
    clean = real[real_indices]
    generated = fake[fake_indices]
    return (
        clean,
        generated,
        damage_slices(clean),
        _shuffle_slices(clean),
    )


@torch.no_grad()
def evaluate_critic(
    critic: LatentCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    damaged: torch.Tensor,
    shuffled: torch.Tensor,
) -> dict[str, torch.Tensor]:
    real_scores = _sample_scores(critic(real))
    fake_scores = _sample_scores(critic(fake))
    damage_scores = _sample_scores(critic(damaged))
    shuffle_scores = _sample_scores(critic(shuffled))
    validation_accuracy = (
        real_scores[:, None] > fake_scores[None, :]
    ).float().mean()
    damage_accuracy = (real_scores > damage_scores).float().mean()
    shuffle_accuracy = (real_scores > shuffle_scores).float().mean()
    score = torch.stack(
        [validation_accuracy, damage_accuracy, shuffle_accuracy]
    ).mean()
    return {
        "validation_margin": real_scores.mean() - fake_scores.mean(),
        "damage_margin": real_scores.mean() - damage_scores.mean(),
        "shuffle_margin": real_scores.mean() - shuffle_scores.mean(),
        "validation_accuracy": validation_accuracy,
        "damage_accuracy": damage_accuracy,
        "shuffle_accuracy": shuffle_accuracy,
        "score": score,
    }


def damage_slices(latents: torch.Tensor) -> torch.Tensor:
    damaged = latents.clone()
    shift = max(1, int(latents.shape[-1]) // 2)
    damaged[:, :, 1::2] = torch.roll(
        damaged[:, :, 1::2],
        shifts=shift,
        dims=-1,
    )
    return damaged


def _sample_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores.flatten(start_dim=1).mean(dim=1)


def _shuffle_slices(latents: torch.Tensor) -> torch.Tensor:
    flat = latents.flatten(start_dim=2)
    order = torch.rand(
        flat.shape[0],
        flat.shape[2],
        device=latents.device,
    ).argsort(dim=1)
    shuffled = flat.gather(2, order.unsqueeze(1).expand_as(flat))
    return shuffled.reshape_as(latents)
