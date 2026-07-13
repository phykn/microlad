from typing import TYPE_CHECKING

import torch

from src.modeling.slicegan import (
    NOISE_CHANNELS,
    SliceGANCritic,
    SliceGANGenerator,
    critic_loss,
    generator_loss,
    gradient_penalty,
)
from src.modeling.slicegan.sampling import sample_slices

if TYPE_CHECKING:
    from src.app.api.options import SliceGANTrainConfig


def train_step(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    anchor_latents: torch.Tensor,
    diffusion_latents: torch.Tensor,
    *,
    noise_size: int,
    mixed: bool,
    config: "SliceGANTrainConfig",
) -> dict[str, torch.Tensor]:
    generator.train()
    critic.train()
    margins = []
    penalties = []
    device = anchor_latents.device
    for _ in range(config.critic_steps):
        optimizer_d.zero_grad(set_to_none=True)
        real = _sample_real(
            anchor_latents,
            diffusion_latents,
            batch_size=config.batch_size,
            mix_probability=config.mix_probability if mixed else 0.0,
        )
        with torch.no_grad():
            fake_volume = generator(
                torch.randn(
                    1,
                    NOISE_CHANNELS,
                    noise_size,
                    noise_size,
                    noise_size,
                    device=device,
                )
            )
            fake = sample_slices(
                fake_volume,
                count=config.batch_size,
                crop_size=int(anchor_latents.shape[-1]),
            )
        real_scores = critic(real)
        fake_scores = critic(fake)
        penalty = gradient_penalty(critic, real, fake)
        loss = critic_loss(
            real_scores,
            fake_scores,
            penalty,
            gradient_weight=config.gradient_weight,
        )
        loss.backward()
        optimizer_d.step()
        margins.append((real_scores.mean() - fake_scores.mean()).detach())
        penalties.append(penalty.detach())

    _set_requires_grad(critic, False)
    optimizer_g.zero_grad(set_to_none=True)
    fake_volume = generator(
        torch.randn(
            1,
            NOISE_CHANNELS,
            noise_size,
            noise_size,
            noise_size,
            device=device,
        )
    )
    fake = sample_slices(
        fake_volume,
        count=config.batch_size,
        crop_size=int(anchor_latents.shape[-1]),
    )
    loss_g = generator_loss(critic(fake))
    loss_g.backward()
    optimizer_g.step()
    _set_requires_grad(critic, True)
    return {
        "slicegan_critic_margin": torch.stack(margins).mean(),
        "slicegan_gradient_penalty": torch.stack(penalties).mean(),
        "slicegan_generator_loss": loss_g.detach(),
    }


def _sample_real(
    anchors: torch.Tensor,
    diffusion: torch.Tensor,
    *,
    batch_size: int,
    mix_probability: float,
) -> torch.Tensor:
    use_diffusion = torch.rand(batch_size, device=anchors.device) < mix_probability
    anchor_indices = torch.randint(
        0,
        anchors.shape[0],
        (batch_size,),
        device=anchors.device,
    )
    diffusion_indices = torch.randint(
        0,
        diffusion.shape[0],
        (batch_size,),
        device=anchors.device,
    )
    selected = torch.where(
        use_diffusion.reshape(-1, 1, 1, 1),
        diffusion[diffusion_indices],
        anchors[anchor_indices],
    )
    augmented = []
    for image in selected:
        turns = int(torch.randint(0, 4, (), device=image.device).item())
        transformed = torch.rot90(image, turns, dims=(-2, -1))
        if bool(torch.randint(0, 2, (), device=image.device).item()):
            transformed = torch.flip(transformed, dims=(-1,))
        shift = torch.randint(0, image.shape[-1], (2,), device=image.device)
        augmented.append(
            torch.roll(transformed, tuple(map(int, shift)), dims=(-2, -1))
        )
    return torch.stack(augmented)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
