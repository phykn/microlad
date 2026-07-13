from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.modeling.phases import probabilities_to_calibrated_labels
from src.modeling.slicegan import (
    SLICEGAN_BASE_NOISE_SIZE,
    SLICEGAN_LATENT_CHANNELS,
    SliceGANCritic,
    SliceGANGenerator,
)
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.conditioning.validation import validate_anchor_intersections
from src.pipelines.guidance.config import (
    SliceGANConditionConfig,
    SliceGANConfig,
    SliceGANRenderConfig,
    SliceGANTrainConfig,
)
from src.pipelines.guidance.descriptors.run_profile import compute_run_profile
from src.pipelines.guidance.diagnostics import phase_volume_diagnostics
from src.pipelines.guidance.slices import (
    critic_slices,
    transition_profile,
    volume_slices,
)
from src.pipelines.reconstruction.volume import decode_latents_with_probabilities
from src.pipelines.scaling.generation import render_generator_tiled


@dataclass(frozen=True)
class _PreparedAnchor:
    labels: torch.Tensor
    probabilities: torch.Tensor
    axis: int
    index: int
    start: int


@dataclass
class _TrainingCandidate:
    step: int
    score: float
    generator: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]


def generate_conditional_slicegan(
    sampler,
    vae: torch.nn.Module,
    *,
    anchors: Sequence[VolumeAnchor],
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    volume_size: int,
    num_phases: int,
    config: SliceGANConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    training = config.training
    conditioning = config.conditioning
    rendering = config.rendering
    seed = config.seed
    _validate_inputs(
        vae,
        anchors=anchors,
        target_fraction=target_fraction,
        phase_fraction_tolerance=phase_fraction_tolerance,
        intersection_tolerance=config.intersection_tolerance,
        volume_size=volume_size,
        num_phases=num_phases,
        steps=training.steps,
        hybrid_steps=training.hybrid_steps,
        condition_steps=conditioning.steps,
        finetune_steps=conditioning.finetune_steps,
        seed=seed,
    )
    prepared_anchors = _prepare_anchors(
        anchors,
        num_phases=num_phases,
        device=device,
    )
    target_fraction = _resolve_target_fraction(
        prepared_anchors,
        target_fraction=target_fraction,
        num_phases=num_phases,
        device=device,
    )
    anchor_references = torch.cat(
        [
            build_anchor_references(anchor.labels, num_phases=num_phases)
            for anchor in prepared_anchors
        ],
        dim=0,
    )

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    _synchronize_device(device)
    total_started = time.perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        reference_started = time.perf_counter()
        _seed_all(seed + 101)
        diffusion_references = build_diffusion_references(
            sampler,
            vae,
            target_fraction=target_fraction,
            num_phases=num_phases,
            count=training.diffusion_reference_count,
        )
        _synchronize_device(device)
        reference_seconds = time.perf_counter() - reference_started

        training_started = time.perf_counter()
        _seed_all(seed)
        generator = SliceGANGenerator(
            num_phases,
            fully_convolutional=volume_size > 64,
        ).to(device)
        critic = SliceGANCritic(num_phases).to(device)
        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=training.learning_rate,
            betas=training.betas,
        )
        optimizer_d = torch.optim.Adam(
            critic.parameters(),
            lr=training.learning_rate,
            betas=training.betas,
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
        candidate_steps = _candidate_steps(training.steps)
        events = sorted(
            set(
                candidate_steps
                + tuple(
                    boundary
                    for boundary in (100, 500, 2000)
                    if boundary < training.steps
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
                stats_prefix="slicegan",
                config=training,
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
            lr=training.learning_rate,
            betas=training.betas,
        )
        hybrid_optimizer_d = torch.optim.Adam(
            critic.parameters(),
            lr=training.learning_rate,
            betas=training.betas,
        )
        hybrid_stats: dict[str, torch.Tensor] = {}
        _replay_training_rng(seed, device=device, num_phases=num_phases)
        hybrid_completed = 0
        while hybrid_completed < training.hybrid_steps:
            segment_steps = min(500, training.hybrid_steps - hybrid_completed)
            hybrid_stats = _train_texture_generator(
                generator,
                critic,
                hybrid_optimizer_g,
                hybrid_optimizer_d,
                hybrid_references,
                steps=segment_steps,
                num_phases=num_phases,
                real_sampler=lambda images, *, batch_size: _sample_hybrid_batch(
                    images,
                    batch_size=batch_size,
                    diffusion_count=int(diffusion_references.shape[0]),
                    diffusion_mix_probability=training.diffusion_mix_probability,
                ),
                stats_prefix="slicegan_hybrid",
                config=training,
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
                    training.steps + hybrid_completed,
                    score,
                    generator,
                    critic,
                )
            )

        _synchronize_device(device)
        training_seconds = time.perf_counter() - training_started

        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.score)
        selected_candidate: _TrainingCandidate | None = None
        selected_quality = torch.full((), float("inf"), device=device)
        selected_volume: torch.Tensor | None = None
        selected_conditioned: torch.Tensor | None = None
        selected_noise_stats: dict[str, torch.Tensor] = {}
        selected_finetune_stats: dict[str, torch.Tensor] = {}
        selected_quality_stats: dict[str, torch.Tensor] = {}
        attempted = 0
        generation_started = time.perf_counter()
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
                prepared_anchors,
                target_transition=morphology_target["transition"],
                volume_size=volume_size,
                steps=conditioning.steps,
                num_phases=num_phases,
                device=device,
                config=conditioning,
                rendering=rendering,
            )
            with torch.no_grad():
                conditioned = probabilities_to_calibrated_labels(
                    _render_inference_probabilities(
                        generator,
                        noise,
                        rendering=rendering,
                    ),
                    num_phases,
                    target_fractions=target_fraction,
                )[0, 0]
            candidate_volume, finetune_stats = _finetune_condition(
                generator,
                critic,
                noise,
                prepared_anchors,
                target_fraction,
                target_transition=morphology_target["transition"],
                steps=conditioning.finetune_steps,
                num_phases=num_phases,
                config=conditioning,
                rendering=rendering,
                optimizer_betas=training.betas,
            )
            quality, quality_stats = _conditional_quality_score(
                candidate_volume,
                morphology_target,
                target_fraction=target_fraction,
                anchors=prepared_anchors,
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
            raise RuntimeError(
                "SliceGAN conditioning did not produce a candidate volume."
            )
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
        _synchronize_device(device)
        generation_seconds = time.perf_counter() - generation_started

    total_seconds = time.perf_counter() - total_started

    stats = {
        "slicegan_steps": torch.tensor(training.steps, device=device),
        "slicegan_hybrid_steps": torch.tensor(training.hybrid_steps, device=device),
        "slicegan_selected_step": torch.tensor(best_step, device=device),
        "slicegan_morphology_score": best_score,
        "slicegan_condition_quality": selected_quality,
        "slicegan_condition_candidates": torch.tensor(attempted, device=device),
        "slicegan_volume_size": torch.tensor(volume_size, device=device),
        "slicegan_fully_convolutional": torch.tensor(
            volume_size > 64,
            device=device,
        ),
        "slicegan_reference_seconds": torch.tensor(
            reference_seconds,
            device=device,
        ),
        "slicegan_training_seconds": torch.tensor(training_seconds, device=device),
        "slicegan_generation_seconds": torch.tensor(
            generation_seconds,
            device=device,
        ),
        "slicegan_total_seconds": torch.tensor(total_seconds, device=device),
        **train_stats,
        **hybrid_stats,
        **noise_stats,
        **finetune_stats,
        **selected_quality_stats,
    }
    stats["slicegan_changed_voxel_fraction"] = (volume != conditioned).float().mean()
    stats["slicegan_phase_fraction"] = torch.stack(
        [(volume == phase).float().mean() for phase in range(num_phases)]
    )
    stats["slicegan_target_phase_fraction"] = target_fraction.detach().clone()
    phase_fraction_error = torch.abs(
        stats["slicegan_phase_fraction"] - stats["slicegan_target_phase_fraction"]
    )
    stats["slicegan_phase_fraction_error"] = phase_fraction_error
    stats["slicegan_phase_fraction_tolerance"] = torch.tensor(
        phase_fraction_tolerance,
        device=device,
    )
    stats["slicegan_phase_fraction_within_tolerance"] = (
        phase_fraction_error.max() <= phase_fraction_tolerance
    )
    if not bool(stats["slicegan_phase_fraction_within_tolerance"].item()):
        raise RuntimeError("generated phase fractions exceed phase_fraction_tolerance.")
    diagnostics = phase_volume_diagnostics(
        volume,
        hybrid_references,
        target_fraction=target_fraction,
        num_phases=num_phases,
    )
    stats.update(
        {f"slicegan_diagnostic_{name}": value for name, value in diagnostics.items()}
    )
    stats["slicegan_anchor_boundary_profile"] = _anchor_boundary_profile(
        volume,
        axis=prepared_anchors[0].axis,
        anchor_index=prepared_anchors[0].index,
    )
    anchor_boundary_stats = [
        _local_boundary_stats(
            volume,
            axis=anchor.axis,
            anchor_index=anchor.index,
        )
        for anchor in prepared_anchors
    ]
    stats["slicegan_anchor_boundary_stds"] = torch.stack(
        [value[0] for value in anchor_boundary_stats]
    )
    stats["slicegan_anchor_boundary_jumps"] = torch.stack(
        [value[1] for value in anchor_boundary_stats]
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
    return (
        F.one_hot(
            labels.long(),
            num_classes=num_phases,
        )
        .movedim(-1, 1)
        .float()
    )


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
    stats_prefix: str,
    config: SliceGANTrainConfig,
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
            SLICEGAN_LATENT_CHANNELS,
            SLICEGAN_BASE_NOISE_SIZE,
            SLICEGAN_BASE_NOISE_SIZE,
            SLICEGAN_BASE_NOISE_SIZE,
            device=device,
        )
        with torch.no_grad():
            fake_volume = generator(noise)

        margins = []
        penalties = []
        for axis in range(3):
            optimizer_d.zero_grad(set_to_none=True)
            real = real_sampler(real_images, batch_size=config.batch_size)
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
            loss = fake_score - real_score + config.gradient_penalty_weight * penalty
            loss.backward()
            optimizer_d.step()
            margins.append((real_score - fake_score).detach())
            penalties.append(penalty.detach())

        if (step + 1) % config.critic_iterations == 0:
            _set_requires_grad(critic, False)
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(
                1,
                SLICEGAN_LATENT_CHANNELS,
                SLICEGAN_BASE_NOISE_SIZE,
                SLICEGAN_BASE_NOISE_SIZE,
                SLICEGAN_BASE_NOISE_SIZE,
                device=device,
            )
            fake_volume = generator(noise)
            generator_loss = sum(
                -critic(
                    critic_slices(
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

    return {
        f"{stats_prefix}_margin": last_margin,
        f"{stats_prefix}_gradient_penalty": last_penalty,
        f"{stats_prefix}_generator_loss": last_generator,
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


def _sample_hybrid_batch(
    images: torch.Tensor,
    *,
    batch_size: int,
    diffusion_count: int,
    diffusion_mix_probability: float,
) -> torch.Tensor:
    if diffusion_count <= 0 or diffusion_count >= images.shape[0]:
        raise ValueError("diffusion_count must split diffusion and anchor references.")
    use_diffusion = (
        torch.rand(batch_size, device=images.device) < diffusion_mix_probability
    )
    diffusion_indices = torch.randint(
        0,
        diffusion_count,
        (batch_size,),
        device=images.device,
    )
    anchor_indices = torch.randint(
        diffusion_count,
        images.shape[0],
        (batch_size,),
        device=images.device,
    )
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
    anchors: Sequence[_PreparedAnchor],
    *,
    target_transition: torch.Tensor,
    volume_size: int,
    steps: int,
    num_phases: int,
    device: torch.device,
    config: SliceGANConditionConfig,
    rendering: SliceGANRenderConfig,
) -> tuple[torch.nn.Parameter, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, False)
    _set_requires_grad(critic, False)
    noise = _select_initial_noise(
        generator,
        anchors,
        noise_size=volume_size // 16,
        device=device,
        candidates=config.noise_candidates,
        rendering=rendering,
    )
    optimizer = torch.optim.Adam([noise], lr=config.noise_lr)
    completed = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = _training_probabilities(generator, noise)
        pixel, shape = _anchor_objective(
            probabilities,
            anchors,
        )
        anchor_transition = _anchor_transition_loss(
            probabilities,
            anchors,
            target_transition=target_transition,
        )
        critic_prior = (
            sum(
                -critic(
                    critic_slices(
                        probabilities,
                        axis,
                        num_phases=num_phases,
                    )
                ).mean()
                for axis in range(3)
            )
            / 3.0
        )
        noise_prior = noise_distribution_loss(noise)
        loss = (
            pixel
            + 2.0 * shape
            + config.anchor_transition_weight * anchor_transition
            + config.noise_critic_weight * critic_prior
            + 5e-2 * noise_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([noise], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            noise.clamp_(-3.5, 3.5)
        completed = step + 1

    with torch.no_grad():
        probabilities = _render_inference_probabilities(
            generator,
            noise,
            rendering=rendering,
        )
        mismatches = _probability_anchor_mismatches(probabilities, anchors)
    return noise, {
        "slicegan_condition_steps": torch.tensor(completed, device=device),
        "slicegan_noise_anchor_mismatch": mismatches.mean(),
        "slicegan_noise_anchor_mismatches": mismatches,
        "slicegan_noise_anchor_max_mismatch": mismatches.max(),
    }


def _select_initial_noise(
    generator: SliceGANGenerator,
    anchors: Sequence[_PreparedAnchor],
    *,
    noise_size: int,
    device: torch.device,
    candidates: int,
    rendering: SliceGANRenderConfig,
) -> torch.nn.Parameter:
    best_noise = None
    best_score = float("inf")
    with torch.no_grad():
        for _ in range(candidates):
            candidate = torch.randn(
                1,
                SLICEGAN_LATENT_CHANNELS,
                noise_size,
                noise_size,
                noise_size,
                device=device,
            )
            probabilities = _render_inference_probabilities(
                generator,
                candidate,
                rendering=rendering,
            )
            _, shape = _anchor_objective(probabilities, anchors)
            score = float(shape.item())
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
    anchors: Sequence[_PreparedAnchor],
    target_fraction: torch.Tensor,
    *,
    target_transition: torch.Tensor,
    steps: int,
    num_phases: int,
    config: SliceGANConditionConfig,
    rendering: SliceGANRenderConfig,
    optimizer_betas: tuple[float, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, True)
    _set_requires_grad(critic, False)
    with torch.no_grad():
        reference_probabilities = _render_inference_probabilities(
            generator,
            noise,
            rendering=rendering,
        ).detach()
    preservation_weights = anchor_preservation_weights(
        reference_probabilities.shape[-3:],
        anchors,
        device=reference_probabilities.device,
        dtype=reference_probabilities.dtype,
        sigma=config.anchor_influence_sigma,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": generator.parameters(), "lr": config.finetune_generator_lr},
            {"params": [noise], "lr": config.finetune_noise_lr},
        ],
        betas=optimizer_betas,
    )
    completed = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = _training_probabilities(generator, noise)
        pixel, shape = _anchor_objective(
            probabilities,
            anchors,
        )
        anchor_transition = _anchor_transition_loss(
            probabilities,
            anchors,
            target_transition=target_transition,
        )
        critic_prior = (
            sum(
                -critic(
                    critic_slices(
                        probabilities,
                        axis,
                        num_phases=num_phases,
                    )
                ).mean()
                for axis in range(3)
            )
            / 3.0
        )
        phase = F.mse_loss(
            probabilities.mean(dim=(0, 2, 3, 4)),
            target_fraction,
        )
        preservation = (
            (probabilities - reference_probabilities).square() * preservation_weights
        ).mean()
        loss = (
            pixel
            + 2.0 * shape
            + config.anchor_transition_weight * anchor_transition
            + config.finetune_critic_weight * critic_prior
            + config.finetune_phase_weight * phase
            + config.finetune_preservation_weight * preservation
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
            mismatches = _probability_anchor_mismatches(
                probabilities,
                anchors,
            )
            if float(mismatches.max().item()) <= config.target_mismatch:
                break

    with torch.no_grad():
        probabilities = _render_inference_probabilities(
            generator,
            noise,
            rendering=rendering,
        )
        volume = probabilities_to_calibrated_labels(
            probabilities,
            num_phases,
            target_fractions=target_fraction,
        )[0, 0]
        mismatches = _label_anchor_mismatches(volume, anchors)
    return volume, {
        "slicegan_finetune_steps": torch.tensor(completed, device=volume.device),
        "slicegan_anchor_mismatch": mismatches.mean(),
        "slicegan_anchor_mismatches": mismatches,
        "slicegan_anchor_max_mismatch": mismatches.max(),
    }


def _render_inference_probabilities(
    generator: SliceGANGenerator,
    noise: torch.Tensor,
    *,
    rendering: SliceGANRenderConfig,
) -> torch.Tensor:
    if max(map(int, noise.shape[-3:])) <= 8:
        return generator(noise)
    return render_generator_tiled(
        generator,
        noise,
        core_noise_size=rendering.core_noise_size,
        halo_noise_size=rendering.halo_noise_size,
        output_device=noise.device,
    )


def _training_probabilities(
    generator: torch.nn.Module,
    noise: torch.Tensor,
) -> torch.Tensor:
    if max(map(int, noise.shape[-3:])) <= 8:
        return generator(noise)
    return checkpoint(generator, noise, use_reentrant=False)


def _categorical_mismatch(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (probabilities.argmax(dim=0) != target).float().mean()


def volume_slice(
    volume: torch.Tensor,
    axis: int,
    index: int,
) -> torch.Tensor:
    if volume.ndim not in (3, 4):
        raise ValueError("volume must have shape [D, H, W] or [C, D, H, W].")
    spatial_axis = axis if volume.ndim == 3 else axis + 1
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if index < 0 or index >= volume.shape[spatial_axis]:
        raise ValueError("index is outside the selected axis.")
    return volume.select(spatial_axis, index)


def _anchor_objective(
    probabilities: torch.Tensor,
    anchors: Sequence[_PreparedAnchor],
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_losses = []
    shape_losses = []
    for anchor in anchors:
        actual = anchor_volume_patch(probabilities[0], anchor)
        pixel_losses.append(
            F.nll_loss(
                actual.clamp_min(1e-8).log().unsqueeze(0),
                anchor.labels.unsqueeze(0),
            )
        )
        shape_losses.append(multiscale_shape_loss(actual, anchor.probabilities))
    return torch.stack(pixel_losses).mean(), torch.stack(shape_losses).mean()


def _anchor_transition_loss(
    probabilities: torch.Tensor,
    anchors: Sequence[_PreparedAnchor],
    *,
    target_transition: torch.Tensor,
) -> torch.Tensor:
    volume = probabilities[0]
    losses = []
    for anchor in anchors:
        transition_rates = transition_profile(volume, anchor.axis)
        start = max(0, anchor.index - 5)
        stop = min(int(transition_rates.shape[0]), anchor.index + 5)
        local = transition_rates[start:stop]
        losses.append(F.mse_loss(local, target_transition.expand_as(local)))
    return torch.stack(losses).mean()


def anchor_preservation_weights(
    spatial_shape: Sequence[int],
    anchors: Sequence[_PreparedAnchor | VolumeAnchor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float,
) -> torch.Tensor:
    shape = tuple(int(size) for size in spatial_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("spatial_shape must contain three positive values.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite.")
    influence = torch.zeros(shape, device=device, dtype=dtype)
    coordinates = torch.meshgrid(
        *[torch.arange(size, device=device, dtype=dtype) for size in shape],
        indexing="ij",
    )
    for anchor in anchors:
        image = anchor.image if isinstance(anchor, VolumeAnchor) else anchor.labels
        patch_size = int(image.shape[-1])
        distance_squared = (coordinates[anchor.axis] - float(anchor.index)).square()
        for dimension in range(3):
            if dimension == anchor.axis:
                continue
            lower = F.relu(float(anchor.start) - coordinates[dimension])
            upper = F.relu(
                coordinates[dimension] - float(anchor.start + patch_size - 1)
            )
            distance_squared = distance_squared + (lower + upper).square()
        local_influence = torch.exp(-0.5 * distance_squared / float(sigma**2))
        influence = torch.maximum(influence, local_influence)
    return (1.0 - influence).reshape(1, 1, *shape)


def _probability_anchor_mismatches(
    probabilities: torch.Tensor,
    anchors: Sequence[_PreparedAnchor],
) -> torch.Tensor:
    return torch.stack(
        [
            _categorical_mismatch(
                anchor_volume_patch(probabilities[0], anchor),
                anchor.labels,
            )
            for anchor in anchors
        ]
    )


def _label_anchor_mismatches(
    volume: torch.Tensor,
    anchors: Sequence[_PreparedAnchor],
) -> torch.Tensor:
    return torch.stack(
        [
            (anchor_volume_patch(volume, anchor) != anchor.labels).float().mean()
            for anchor in anchors
        ]
    )


def anchor_volume_patch(
    volume: torch.Tensor,
    anchor: _PreparedAnchor | VolumeAnchor,
) -> torch.Tensor:
    plane = volume_slice(volume, anchor.axis, anchor.index)
    image = anchor.image if isinstance(anchor, VolumeAnchor) else anchor.labels
    height, width = map(int, image.shape)
    stop_row = anchor.start + height
    stop_col = anchor.start + width
    if anchor.start < 0 or stop_row > plane.shape[-2] or stop_col > plane.shape[-1]:
        raise ValueError("anchor patch is outside the selected volume slice.")
    return plane[..., anchor.start : stop_row, anchor.start : stop_col]


def _anchor_boundary_profile(
    volume: torch.Tensor,
    *,
    axis: int,
    anchor_index: int,
    radius: int = 10,
) -> torch.Tensor:
    length = int(volume.shape[axis])
    rates = (
        (volume.narrow(axis, 1, length - 1) != volume.narrow(axis, 0, length - 1))
        .float()
        .mean(dim=tuple(dimension for dimension in range(3) if dimension != axis))
    )
    start = max(0, anchor_index - radius)
    stop = min(int(rates.shape[0]), anchor_index + radius)
    return rates[start:stop]


def _local_boundary_stats(
    volume: torch.Tensor,
    *,
    axis: int,
    anchor_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile = _anchor_boundary_profile(
        volume,
        axis=axis,
        anchor_index=anchor_index,
        radius=5,
    )
    boundary_std = profile.std(unbiased=False)
    boundary_jump = (
        torch.abs(profile[1:] - profile[:-1]).max()
        if profile.numel() > 1
        else profile.new_zeros(())
    )
    return boundary_std, boundary_jump


def _prepare_anchors(
    anchors: Sequence[VolumeAnchor],
    *,
    num_phases: int,
    device: torch.device,
) -> list[_PreparedAnchor]:
    prepared = []
    for anchor in anchors:
        labels = anchor.image.to(device=device, dtype=torch.long)
        probabilities = (
            F.one_hot(
                labels,
                num_classes=num_phases,
            )
            .movedim(-1, 0)
            .float()
        )
        prepared.append(
            _PreparedAnchor(
                labels=labels,
                probabilities=probabilities,
                axis=int(anchor.axis),
                index=int(anchor.index),
                start=int(anchor.start),
            )
        )
    return prepared


def _resolve_target_fraction(
    anchors: Sequence[_PreparedAnchor],
    *,
    target_fraction: torch.Tensor | None,
    num_phases: int,
    device: torch.device,
) -> torch.Tensor:
    if target_fraction is not None:
        return torch.as_tensor(
            target_fraction,
            device=device,
            dtype=torch.float32,
        )
    return (
        torch.stack([anchor.probabilities.mean(dim=(1, 2)) for anchor in anchors])
        .mean(dim=0)
        .reshape(num_phases)
    )


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _candidate_steps(steps: int) -> tuple[int, ...]:
    return (steps,)


def _consume_fixed_noise(device: torch.device) -> torch.Tensor:
    return torch.randn(
        1,
        SLICEGAN_LATENT_CHANNELS,
        SLICEGAN_BASE_NOISE_SIZE,
        SLICEGAN_BASE_NOISE_SIZE,
        SLICEGAN_BASE_NOISE_SIZE,
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
    probabilities = (
        F.one_hot(
            labels.unsqueeze(0),
            num_classes=num_phases,
        )
        .movedim(-1, 1)
        .float()
    )
    phase = probabilities.mean(dim=(0, 2, 3, 4))
    phase_error = torch.mean(torch.abs(phase - target_fraction))
    transitions = torch.stack(
        [
            (
                labels.narrow(dimension, 1, labels.shape[dimension] - 1)
                != labels.narrow(dimension, 0, labels.shape[dimension] - 1)
            )
            .float()
            .mean()
            for dimension in (0, 1, 2)
        ]
    )
    transition_error = torch.mean(torch.abs(transitions - target["transition"]))
    run_profile = compute_run_profile(
        probabilities,
        lengths=(2, 4, 8, 16),
    )
    run_error = torch.mean(torch.abs(run_profile - target["run_profile"].unsqueeze(0)))
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
    anchors: Sequence[_PreparedAnchor],
    num_phases: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    morphology, phase_error, transition_error, run_error = _volume_morphology_errors(
        volume,
        target,
        target_fraction=target_fraction,
        num_phases=num_phases,
    )
    mismatches = _label_anchor_mismatches(volume, anchors)
    mismatch = mismatches.mean()
    max_mismatch = mismatches.max()
    boundary_values = [
        _local_boundary_stats(
            volume,
            axis=anchor.axis,
            anchor_index=anchor.index,
        )
        for anchor in anchors
    ]
    boundary_std = torch.stack([value[0] for value in boundary_values]).max()
    boundary_jump = torch.stack([value[1] for value in boundary_values]).max()
    mismatch_penalty = 10.0 * F.relu(max_mismatch - 0.10)
    quality = (
        morphology + 0.2 * mismatch + boundary_std + boundary_jump + mismatch_penalty
    )
    return quality, {
        "slicegan_quality_anchor_mismatch": mismatch,
        "slicegan_quality_anchor_mismatches": mismatches,
        "slicegan_quality_anchor_max_mismatch": max_mismatch,
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
        float(stats["slicegan_quality_anchor_max_mismatch"].item()) <= 0.10
        and float(stats["slicegan_quality_phase_mae"].item()) <= 0.01
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
    return {name: value.detach().clone() for name, value in stats.items()}


def _validate_inputs(
    vae: torch.nn.Module,
    *,
    anchors: Sequence[VolumeAnchor],
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    intersection_tolerance: float,
    volume_size: int,
    num_phases: int,
    steps: int,
    hybrid_steps: int,
    condition_steps: int,
    finetune_steps: int,
    seed: int,
) -> None:
    if int(vae.image_size) != 64:
        raise ValueError("conditional SliceGAN currently requires vae.image_size=64.")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if (
        not np.isfinite(phase_fraction_tolerance)
        or phase_fraction_tolerance < 0.0
        or phase_fraction_tolerance > 1.0
    ):
        raise ValueError("phase_fraction_tolerance must be between 0 and 1.")
    if (
        not isinstance(volume_size, int)
        or isinstance(volume_size, bool)
        or volume_size < 64
        or volume_size % 64 != 0
    ):
        raise ValueError("volume_size must be a positive multiple of 64.")
    if not anchors:
        raise ValueError("conditional SliceGAN requires at least one anchor.")
    for anchor in anchors:
        if not isinstance(anchor, VolumeAnchor):
            raise TypeError("anchors must contain VolumeAnchor values.")
        if anchor.image.shape != torch.Size((64, 64)):
            raise ValueError("SliceGAN anchor image must have shape [64, 64].")
        if anchor.axis not in (0, 1, 2):
            raise ValueError("SliceGAN anchor axis must be 0, 1, or 2.")
        if anchor.index < 0 or anchor.index >= volume_size:
            raise ValueError("SliceGAN anchor index is outside volume_size.")
        if anchor.start < 0 or anchor.start + 64 > volume_size:
            raise ValueError("SliceGAN anchor patch is outside volume_size.")
        if not torch.isfinite(anchor.image).all():
            raise ValueError("SliceGAN anchor image must be finite.")
        if not torch.equal(anchor.image, anchor.image.round()):
            raise ValueError("SliceGAN anchor image must contain categorical labels.")
        if anchor.image.min().item() < 0 or anchor.image.max().item() >= num_phases:
            raise ValueError("SliceGAN anchor labels must be inside the phase range.")
    validate_anchor_intersections(
        anchors,
        tolerance=intersection_tolerance,
    )
    if target_fraction is not None:
        fractions = torch.as_tensor(target_fraction)
        if fractions.shape != (num_phases,):
            raise ValueError("target_fraction must have shape [num_phases].")
        if not torch.isfinite(fractions).all() or torch.any(fractions < 0.0):
            raise ValueError("target_fraction must be finite and non-negative.")
        if not torch.allclose(
            fractions.sum(), torch.ones_like(fractions.sum()), atol=1e-4
        ):
            raise ValueError("target_fraction must sum to one.")
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
