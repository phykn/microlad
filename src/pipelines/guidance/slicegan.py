from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.modeling.phases import probabilities_to_calibrated_labels
from src.pipelines.guidance.descriptors.run_profile import compute_run_profile
from src.pipelines.reconstruction.volume import decode_latents_with_probabilities


_LATENT_CHANNELS = 32
_LATENT_SIZE = 4
_CRITIC_ITERS = 5
_BATCH_SIZE = 8
_LEARNING_RATE = 1e-4
_BETAS = (0.9, 0.99)
_GRADIENT_PENALTY_WEIGHT = 10.0
_DIFFUSION_REFERENCE_COUNT = 8
_DIFFUSION_MIX_PROBABILITY = 0.1
_NOISE_CANDIDATES = 8
_NOISE_LR = 5e-2
_NOISE_CRITIC_WEIGHT = 1e-2
_FINETUNE_GENERATOR_LR = 1e-5
_FINETUNE_NOISE_LR = 2e-3
_FINETUNE_CRITIC_WEIGHT = 2e-2
_FINETUNE_PHASE_WEIGHT = 50.0
_TARGET_MISMATCH = 0.08


@dataclass
class _TrainingCandidate:
    step: int
    score: float
    generator: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]


class SliceGANGenerator(torch.nn.Module):
    def __init__(self, num_phases: int) -> None:
        super().__init__()
        channels = (_LATENT_CHANNELS, 1024, 512, 128, 32)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(
                        source,
                        target,
                        kernel_size=4,
                        stride=2,
                        padding=2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm3d(target),
                    torch.nn.ReLU(inplace=True),
                )
                for source, target in zip(channels, channels[1:])
            ]
        )
        self.to_logits = torch.nn.Conv3d(
            channels[-1],
            num_phases,
            kernel_size=3,
            padding=0,
            bias=False,
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = noise
        for block in self.blocks:
            x = block(x)
        x = F.interpolate(
            x,
            size=(66, 66, 66),
            mode="trilinear",
            align_corners=False,
        )
        return torch.softmax(self.to_logits(x), dim=1)


class SliceGANCritic(torch.nn.Module):
    def __init__(self, num_phases: int) -> None:
        super().__init__()
        channels = (num_phases, 64, 128, 256, 512, 1)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    source,
                    target,
                    kernel_size=4,
                    stride=2,
                    padding=1 if index < 4 else 0,
                    bias=False,
                )
                for index, (source, target) in enumerate(
                    zip(channels, channels[1:])
                )
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def generate_conditional_slicegan(
    sampler,
    vae: torch.nn.Module,
    *,
    anchor_image: torch.Tensor,
    anchor_index: int,
    num_phases: int,
    steps: int,
    hybrid_steps: int,
    condition_steps: int,
    finetune_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_inputs(
        vae,
        anchor_image=anchor_image,
        anchor_index=anchor_index,
        num_phases=num_phases,
        steps=steps,
        hybrid_steps=hybrid_steps,
        condition_steps=condition_steps,
        finetune_steps=finetune_steps,
        seed=seed,
    )
    anchor = anchor_image.to(device=device, dtype=torch.long)
    target_one_hot = F.one_hot(
        anchor,
        num_classes=num_phases,
    ).movedim(-1, 0).float()
    target_fraction = target_one_hot.mean(dim=(1, 2))
    anchor_references = build_anchor_references(
        anchor,
        num_phases=num_phases,
    )

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        _seed_all(seed + 101)
        diffusion_references = build_diffusion_references(
            sampler,
            vae,
            target_fraction=target_fraction,
            num_phases=num_phases,
            count=_DIFFUSION_REFERENCE_COUNT,
        )

        _seed_all(seed)
        generator = SliceGANGenerator(num_phases).to(device)
        critic = SliceGANCritic(num_phases).to(device)
        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=_LEARNING_RATE,
            betas=_BETAS,
        )
        optimizer_d = torch.optim.Adam(
            critic.parameters(),
            lr=_LEARNING_RATE,
            betas=_BETAS,
        )
        fixed_noise = _consume_fixed_noise(device)
        hybrid_references = torch.cat(
            [diffusion_references, anchor_references],
            dim=0,
        )
        morphology_target = _build_morphology_target(
            hybrid_references,
            target_fraction=target_fraction,
        )
        train_stats: dict[str, torch.Tensor] = {}
        candidates: list[_TrainingCandidate] = []
        current_step = 0
        candidate_steps = _candidate_steps(steps)
        events = sorted(
            set(
                candidate_steps
                + tuple(
                    boundary
                    for boundary in (100, 500, 2000)
                    if boundary < steps
                )
            )
        )
        for event in events:
            if current_step in (100, 500, 2000):
                _replay_training_rng(
                    seed,
                    device=device,
                    num_phases=num_phases,
                )
            train_stats = _train_texture_generator(
                generator,
                critic,
                optimizer_g,
                optimizer_d,
                anchor_references,
                steps=event - current_step,
                num_phases=num_phases,
                real_sampler=_sample_anchor_batch,
            )
            current_step = event
            if event in candidate_steps:
                score = _morphology_score(
                    generator,
                    fixed_noise,
                    morphology_target,
                    target_fraction=target_fraction,
                    num_phases=num_phases,
                )
                candidates.append(
                    _capture_candidate(
                        event,
                        score,
                        generator,
                        critic,
                    )
                )

        if not candidates:
            raise RuntimeError("SliceGAN training did not produce a candidate state.")
        primary_candidate = min(candidates, key=lambda candidate: candidate.score)
        generator.load_state_dict(primary_candidate.generator)
        critic.load_state_dict(primary_candidate.critic)

        hybrid_optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=_LEARNING_RATE,
            betas=_BETAS,
        )
        hybrid_optimizer_d = torch.optim.Adam(
            critic.parameters(),
            lr=_LEARNING_RATE,
            betas=_BETAS,
        )
        hybrid_stats: dict[str, torch.Tensor] = {}
        _replay_training_rng(seed, device=device, num_phases=num_phases)
        hybrid_completed = 0
        while hybrid_completed < hybrid_steps:
            segment_steps = min(500, hybrid_steps - hybrid_completed)
            hybrid_stats = _train_texture_generator(
                generator,
                critic,
                hybrid_optimizer_g,
                hybrid_optimizer_d,
                hybrid_references,
                steps=segment_steps,
                num_phases=num_phases,
                real_sampler=_sample_hybrid_batch,
            )
            hybrid_completed += segment_steps
            score = _morphology_score(
                generator,
                fixed_noise,
                morphology_target,
                target_fraction=target_fraction,
                num_phases=num_phases,
            )
            candidates.append(
                _capture_candidate(
                    steps + hybrid_completed,
                    score,
                    generator,
                    critic,
                )
            )

        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.score)
        selected_candidate: _TrainingCandidate | None = None
        selected_quality = torch.full((), float("inf"), device=device)
        selected_volume: torch.Tensor | None = None
        selected_conditioned: torch.Tensor | None = None
        selected_noise_stats: dict[str, torch.Tensor] = {}
        selected_finetune_stats: dict[str, torch.Tensor] = {}
        selected_quality_stats: dict[str, torch.Tensor] = {}
        attempted = 0
        for candidate in ordered_candidates:
            if attempted >= 3 and _quality_passes(selected_quality_stats):
                break
            attempted += 1
            generator.load_state_dict(candidate.generator)
            critic.load_state_dict(candidate.critic)
            _seed_all(seed + 1)
            noise, noise_stats = _condition_noise(
                generator,
                critic,
                target_one_hot,
                anchor_index=anchor_index,
                steps=condition_steps,
                num_phases=num_phases,
                device=device,
            )
            with torch.no_grad():
                conditioned = generator(noise).argmax(dim=1)[0]
            candidate_volume, finetune_stats = _finetune_condition(
                generator,
                critic,
                noise,
                target_one_hot,
                target_fraction,
                anchor_index=anchor_index,
                steps=finetune_steps,
                num_phases=num_phases,
            )
            quality, quality_stats = _conditional_quality_score(
                candidate_volume,
                morphology_target,
                target_fraction=target_fraction,
                target_labels=anchor,
                anchor_index=anchor_index,
                num_phases=num_phases,
            )
            if bool((quality < selected_quality).item()):
                selected_candidate = candidate
                selected_quality = quality
                selected_volume = candidate_volume.detach().clone()
                selected_conditioned = conditioned.detach().clone()
                selected_noise_stats = _clone_tensor_stats(noise_stats)
                selected_finetune_stats = _clone_tensor_stats(finetune_stats)
                selected_quality_stats = _clone_tensor_stats(quality_stats)

        if (
            selected_candidate is None
            or selected_volume is None
            or selected_conditioned is None
        ):
            raise RuntimeError("SliceGAN conditioning did not produce a candidate volume.")
        volume = selected_volume
        conditioned = selected_conditioned
        noise_stats = selected_noise_stats
        finetune_stats = selected_finetune_stats
        best_step = selected_candidate.step
        best_score = torch.tensor(
            selected_candidate.score,
            device=device,
            dtype=torch.float32,
        )

    stats = {
        "slicegan_steps": torch.tensor(steps, device=device),
        "slicegan_hybrid_steps": torch.tensor(hybrid_steps, device=device),
        "slicegan_selected_step": torch.tensor(best_step, device=device),
        "slicegan_morphology_score": best_score,
        "slicegan_condition_quality": selected_quality,
        "slicegan_condition_candidates": torch.tensor(attempted, device=device),
        **train_stats,
        **hybrid_stats,
        **noise_stats,
        **finetune_stats,
        **selected_quality_stats,
    }
    stats["slicegan_changed_voxel_fraction"] = (
        volume != conditioned
    ).float().mean()
    stats["slicegan_phase_fraction"] = torch.stack(
        [(volume == phase).float().mean() for phase in range(num_phases)]
    )
    stats["slicegan_anchor_boundary_profile"] = _anchor_boundary_profile(
        volume,
        anchor_index=anchor_index,
    )
    return volume.float(), stats


@torch.no_grad()
def build_diffusion_references(
    sampler,
    vae: torch.nn.Module,
    *,
    target_fraction: torch.Tensor,
    num_phases: int,
    count: int,
) -> torch.Tensor:
    vae.eval()
    latents = sampler.sample(
        (
            count,
            int(vae.latent_ch),
            int(vae.latent_size),
            int(vae.latent_size),
        )
    )
    _, probabilities = decode_latents_with_probabilities(
        vae,
        latents,
        num_phases=num_phases,
    )
    if probabilities is None:
        raise ValueError("categorical VAE probabilities are required for SliceGAN.")
    labels = probabilities_to_calibrated_labels(
        probabilities,
        num_phases,
        target_fractions=target_fraction,
    )[:, 0]
    return F.one_hot(
        labels.long(),
        num_classes=num_phases,
    ).movedim(-1, 1).float()


def build_anchor_references(
    anchor: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    images = []
    for flipped in (anchor, torch.flip(anchor, dims=(-1,))):
        images.extend(torch.rot90(flipped, turns, dims=(-2, -1)) for turns in range(4))
    labels = torch.stack(images)
    return F.one_hot(labels, num_classes=num_phases).movedim(-1, 1).float()


def volume_slices(
    volume: torch.Tensor,
    axis: int,
    *,
    num_phases: int,
) -> torch.Tensor:
    if axis == 0:
        return volume.permute(0, 2, 1, 3, 4).reshape(-1, num_phases, 64, 64)
    if axis == 1:
        return volume.permute(0, 3, 1, 2, 4).reshape(-1, num_phases, 64, 64)
    if axis == 2:
        return volume.permute(0, 4, 1, 2, 3).reshape(-1, num_phases, 64, 64)
    raise ValueError("axis must be 0, 1, or 2.")


def multiscale_shape_loss(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    actual_batch = actual.unsqueeze(0)
    target_batch = target.unsqueeze(0)
    losses = [
        F.mse_loss(
            F.avg_pool2d(actual_batch, kernel_size=scale, stride=scale),
            F.avg_pool2d(target_batch, kernel_size=scale, stride=scale),
        )
        for scale in (2, 4, 8)
    ]
    return torch.stack(losses).mean()


def noise_distribution_loss(noise: torch.Tensor) -> torch.Tensor:
    mean = noise.mean()
    std = noise.std(unbiased=False)
    tail = F.relu(noise.abs() - 3.0).square().mean()
    return mean.square() + (std - 1.0).square() + tail


def _train_texture_generator(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    real_images: torch.Tensor,
    *,
    steps: int,
    num_phases: int,
    real_sampler: Callable[..., torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = real_images.device
    last_margin = torch.zeros((), device=device)
    last_penalty = torch.zeros((), device=device)
    last_generator = torch.zeros((), device=device)
    for step in range(steps):
        generator.train()
        critic.train()
        noise = torch.randn(
            1,
            _LATENT_CHANNELS,
            _LATENT_SIZE,
            _LATENT_SIZE,
            _LATENT_SIZE,
            device=device,
        )
        with torch.no_grad():
            fake_volume = generator(noise)

        margins = []
        penalties = []
        for axis in range(3):
            optimizer_d.zero_grad(set_to_none=True)
            real = real_sampler(real_images, batch_size=_BATCH_SIZE)
            fake = volume_slices(
                fake_volume,
                axis,
                num_phases=num_phases,
            )
            real_score = critic(real).mean()
            fake_score = critic(fake).mean()
            penalty = _gradient_penalty(
                critic,
                real,
                fake[: real.shape[0]],
            )
            loss = (
                fake_score
                - real_score
                + _GRADIENT_PENALTY_WEIGHT * penalty
            )
            loss.backward()
            optimizer_d.step()
            margins.append((real_score - fake_score).detach())
            penalties.append(penalty.detach())

        if (step + 1) % _CRITIC_ITERS == 0:
            _set_requires_grad(critic, False)
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(
                1,
                _LATENT_CHANNELS,
                _LATENT_SIZE,
                _LATENT_SIZE,
                _LATENT_SIZE,
                device=device,
            )
            fake_volume = generator(noise)
            generator_loss = sum(
                -critic(
                    volume_slices(
                        fake_volume,
                        axis,
                        num_phases=num_phases,
                    )
                ).mean()
                for axis in range(3)
            )
            generator_loss.backward()
            optimizer_g.step()
            last_generator = generator_loss.detach()
            _set_requires_grad(critic, True)

        last_margin = torch.stack(margins).mean()
        last_penalty = torch.stack(penalties).mean()

    prefix = "slicegan_hybrid" if real_sampler is _sample_hybrid_batch else "slicegan"
    return {
        f"{prefix}_margin": last_margin,
        f"{prefix}_gradient_penalty": last_penalty,
        f"{prefix}_generator_loss": last_generator,
    }


def _sample_anchor_batch(images: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    indices = torch.randint(0, images.shape[0], (batch_size,), device=images.device)
    selected = images[indices]
    shifts = torch.randint(0, images.shape[-1], (batch_size, 2), device=images.device)
    return torch.stack(
        [
            torch.roll(image, tuple(map(int, shift)), dims=(-2, -1))
            for image, shift in zip(selected, shifts)
        ]
    )


def _sample_hybrid_batch(images: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    use_diffusion = torch.rand(batch_size, device=images.device) < _DIFFUSION_MIX_PROBABILITY
    diffusion_indices = torch.randint(0, 8, (batch_size,), device=images.device)
    anchor_indices = torch.randint(8, 16, (batch_size,), device=images.device)
    indices = torch.where(use_diffusion, diffusion_indices, anchor_indices)
    selected = images[indices]
    augmented = []
    for image, is_diffusion in zip(selected, use_diffusion):
        if bool(is_diffusion.item()):
            turns = int(torch.randint(0, 4, (), device=images.device).item())
            transformed = torch.rot90(image, turns, dims=(-2, -1))
            if bool(torch.randint(0, 2, (), device=images.device).item()):
                transformed = torch.flip(transformed, dims=(-1,))
        else:
            shift = torch.randint(0, image.shape[-1], (2,), device=images.device)
            transformed = torch.roll(
                image,
                tuple(map(int, shift)),
                dims=(-2, -1),
            )
        augmented.append(transformed)
    return torch.stack(augmented)


def _gradient_penalty(
    critic: SliceGANCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    mixed = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)
    gradients = torch.autograd.grad(
        outputs=critic(mixed).sum(),
        inputs=mixed,
        create_graph=True,
        only_inputs=True,
    )[0]
    norm = gradients.flatten(start_dim=1).norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def _condition_noise(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    target: torch.Tensor,
    *,
    anchor_index: int,
    steps: int,
    num_phases: int,
    device: torch.device,
) -> tuple[torch.nn.Parameter, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, False)
    _set_requires_grad(critic, False)
    noise = _select_initial_noise(
        generator,
        target,
        anchor_index=anchor_index,
        device=device,
    )
    optimizer = torch.optim.Adam([noise], lr=_NOISE_LR)
    target_labels = target.argmax(dim=0)
    completed = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = generator(noise)
        center = probabilities[0, :, anchor_index]
        pixel = F.nll_loss(
            center.clamp_min(1e-8).log().unsqueeze(0),
            target_labels.unsqueeze(0),
        )
        shape = multiscale_shape_loss(center, target)
        critic_prior = sum(
            -critic(
                volume_slices(
                    probabilities,
                    axis,
                    num_phases=num_phases,
                )
            ).mean()
            for axis in range(3)
        ) / 3.0
        noise_prior = noise_distribution_loss(noise)
        loss = (
            pixel
            + 2.0 * shape
            + _NOISE_CRITIC_WEIGHT * critic_prior
            + 5e-2 * noise_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([noise], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            noise.clamp_(-3.5, 3.5)
        completed = step + 1

    with torch.no_grad():
        probabilities = generator(noise)
        mismatch = _categorical_mismatch(
            probabilities[0, :, anchor_index],
            target_labels,
        )
    return noise, {
        "slicegan_condition_steps": torch.tensor(completed, device=device),
        "slicegan_noise_anchor_mismatch": mismatch,
    }


def _select_initial_noise(
    generator: SliceGANGenerator,
    target: torch.Tensor,
    *,
    anchor_index: int,
    device: torch.device,
) -> torch.nn.Parameter:
    best_noise = None
    best_score = float("inf")
    with torch.no_grad():
        for _ in range(_NOISE_CANDIDATES):
            candidate = torch.randn(
                1,
                _LATENT_CHANNELS,
                _LATENT_SIZE,
                _LATENT_SIZE,
                _LATENT_SIZE,
                device=device,
            )
            score = float(
                multiscale_shape_loss(
                    generator(candidate)[0, :, anchor_index],
                    target,
                ).item()
            )
            if score < best_score:
                best_score = score
                best_noise = candidate.detach().clone()
    if best_noise is None:
        raise RuntimeError("failed to initialize SliceGAN conditioning noise.")
    return torch.nn.Parameter(best_noise)


def _finetune_condition(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    noise: torch.nn.Parameter,
    target: torch.Tensor,
    target_fraction: torch.Tensor,
    *,
    anchor_index: int,
    steps: int,
    num_phases: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, True)
    _set_requires_grad(critic, False)
    optimizer = torch.optim.Adam(
        [
            {"params": generator.parameters(), "lr": _FINETUNE_GENERATOR_LR},
            {"params": [noise], "lr": _FINETUNE_NOISE_LR},
        ],
        betas=_BETAS,
    )
    target_labels = target.argmax(dim=0)
    completed = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = generator(noise)
        center = probabilities[0, :, anchor_index]
        pixel = F.nll_loss(
            center.clamp_min(1e-8).log().unsqueeze(0),
            target_labels.unsqueeze(0),
        )
        shape = multiscale_shape_loss(center, target)
        critic_prior = sum(
            -critic(
                volume_slices(
                    probabilities,
                    axis,
                    num_phases=num_phases,
                )
            ).mean()
            for axis in range(3)
        ) / 3.0
        phase = F.mse_loss(
            probabilities.mean(dim=(0, 2, 3, 4)),
            target_fraction,
        )
        loss = (
            pixel
            + 2.0 * shape
            + _FINETUNE_CRITIC_WEIGHT * critic_prior
            + _FINETUNE_PHASE_WEIGHT * phase
            + 5e-2 * noise_distribution_loss(noise)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([noise], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            noise.clamp_(-3.5, 3.5)
        completed = step + 1
        if completed % 10 == 0:
            mismatch = _categorical_mismatch(center, target_labels)
            if float(mismatch.item()) <= _TARGET_MISMATCH:
                break

    with torch.no_grad():
        probabilities = generator(noise)
        volume = probabilities.argmax(dim=1)[0]
        mismatch = _categorical_mismatch(
            probabilities[0, :, anchor_index],
            target_labels,
        )
    return volume, {
        "slicegan_finetune_steps": torch.tensor(completed, device=volume.device),
        "slicegan_anchor_mismatch": mismatch,
    }


def _categorical_mismatch(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (probabilities.argmax(dim=0) != target).float().mean()


def _anchor_boundary_profile(
    volume: torch.Tensor,
    *,
    anchor_index: int,
) -> torch.Tensor:
    rates = (volume[1:] != volume[:-1]).float().mean(dim=(1, 2))
    start = max(0, anchor_index - 10)
    stop = min(int(rates.shape[0]), anchor_index + 10)
    return rates[start:stop]


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _candidate_steps(steps: int) -> tuple[int, ...]:
    candidates = [step for step in (3000, 4000) if step < steps]
    candidates.append(steps)
    return tuple(candidates)


def _consume_fixed_noise(device: torch.device) -> torch.Tensor:
    return torch.randn(
        1,
        _LATENT_CHANNELS,
        _LATENT_SIZE,
        _LATENT_SIZE,
        _LATENT_SIZE,
        device=device,
    )


def _replay_training_rng(
    seed: int,
    *,
    device: torch.device,
    num_phases: int,
) -> None:
    _seed_all(seed)
    # Checkpointed experiments restarted the process before each segment. Model
    # construction and the fixed preview noise consumed RNG values before the
    # saved weights and optimizer state were restored. Replaying those draws
    # keeps the in-memory path deterministic without writing 500 MB checkpoints.
    SliceGANGenerator(num_phases)
    SliceGANCritic(num_phases)
    _consume_fixed_noise(device)


def _build_morphology_target(
    references: torch.Tensor,
    *,
    target_fraction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    labels = references.argmax(dim=1)
    transition = 0.5 * (
        (labels[:, :, 1:] != labels[:, :, :-1]).float().mean()
        + (labels[:, 1:, :] != labels[:, :-1, :]).float().mean()
    )
    run_profile = compute_run_profile(
        references,
        lengths=(2, 4, 8, 16),
    ).mean(dim=0)
    return {
        "phase_fraction": target_fraction,
        "transition": transition,
        "run_profile": run_profile,
    }


@torch.no_grad()
def _morphology_score(
    generator: SliceGANGenerator,
    fixed_noise: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    target_fraction: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    generator.eval()
    labels = generator(fixed_noise).argmax(dim=1)[0]
    score, _, _, _ = _volume_morphology_errors(
        labels,
        target,
        target_fraction=target_fraction,
        num_phases=num_phases,
    )
    generator.train()
    return score


@torch.no_grad()
def _volume_morphology_errors(
    labels: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    target_fraction: torch.Tensor,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.one_hot(
        labels.unsqueeze(0),
        num_classes=num_phases,
    ).movedim(-1, 1).float()
    phase = probabilities.mean(dim=(0, 2, 3, 4))
    phase_error = torch.mean(torch.abs(phase - target_fraction))
    transitions = torch.stack(
        [
            (labels.narrow(dimension, 1, 63) != labels.narrow(dimension, 0, 63))
            .float()
            .mean()
            for dimension in (0, 1, 2)
        ]
    )
    transition_error = torch.mean(
        torch.abs(transitions - target["transition"])
    )
    run_profile = compute_run_profile(
        probabilities,
        lengths=(2, 4, 8, 16),
    )
    run_error = torch.mean(
        torch.abs(run_profile - target["run_profile"].unsqueeze(0))
    )
    return (
        phase_error + transition_error + run_error,
        phase_error,
        transition_error,
        run_error,
    )


@torch.no_grad()
def _conditional_quality_score(
    volume: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    target_fraction: torch.Tensor,
    target_labels: torch.Tensor,
    anchor_index: int,
    num_phases: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    morphology, phase_error, transition_error, run_error = _volume_morphology_errors(
        volume,
        target,
        target_fraction=target_fraction,
        num_phases=num_phases,
    )
    mismatch = (volume[anchor_index] != target_labels).float().mean()
    rates = (volume[1:] != volume[:-1]).float().mean(dim=(1, 2))
    start = max(0, anchor_index - 5)
    stop = min(int(rates.shape[0]), anchor_index + 5)
    local = rates[start:stop]
    boundary_std = local.std(unbiased=False)
    boundary_jump = (
        torch.abs(local[1:] - local[:-1]).max()
        if local.numel() > 1
        else local.new_zeros(())
    )
    mismatch_penalty = 10.0 * F.relu(mismatch - 0.085)
    quality = (
        morphology
        + 0.2 * mismatch
        + boundary_std
        + boundary_jump
        + mismatch_penalty
    )
    return quality, {
        "slicegan_quality_anchor_mismatch": mismatch,
        "slicegan_quality_phase_mae": phase_error,
        "slicegan_quality_transition_mae": transition_error,
        "slicegan_quality_run_mae": run_error,
        "slicegan_quality_boundary_std": boundary_std,
        "slicegan_quality_boundary_jump": boundary_jump,
    }


def _quality_passes(stats: dict[str, torch.Tensor]) -> bool:
    if not stats:
        return False
    return (
        float(stats["slicegan_quality_anchor_mismatch"].item()) <= 0.085
        and float(stats["slicegan_quality_phase_mae"].item()) <= 0.015
        and float(stats["slicegan_quality_boundary_std"].item()) <= 0.03
        and float(stats["slicegan_quality_boundary_jump"].item()) <= 0.08
    )


def _capture_candidate(
    step: int,
    score: torch.Tensor,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
) -> _TrainingCandidate:
    return _TrainingCandidate(
        step=step,
        score=float(score.item()),
        generator=_clone_module_state(generator),
        critic=_clone_module_state(critic),
    )


def _clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _clone_tensor_stats(
    stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in stats.items()
    }


def _validate_inputs(
    vae: torch.nn.Module,
    *,
    anchor_image: torch.Tensor,
    anchor_index: int,
    num_phases: int,
    steps: int,
    hybrid_steps: int,
    condition_steps: int,
    finetune_steps: int,
    seed: int,
) -> None:
    if int(vae.image_size) != 64:
        raise ValueError("conditional SliceGAN currently requires vae.image_size=64.")
    if anchor_image.shape != torch.Size((64, 64)):
        raise ValueError("SliceGAN anchor image must have shape [64, 64].")
    if anchor_index < 0 or anchor_index >= 64:
        raise ValueError("SliceGAN anchor index must be between 0 and 63.")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    for name, value in (
        ("steps", steps),
        ("hybrid_steps", hybrid_steps),
        ("condition_steps", condition_steps),
        ("finetune_steps", finetune_steps),
        ("seed", seed),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
