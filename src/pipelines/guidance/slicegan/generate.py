from collections.abc import Sequence
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from src.modeling.slicegan import (
    NOISE_CHANNELS,
    SCALE_FACTOR,
    SliceGANCritic,
    SliceGANGenerator,
)
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.slicegan.anchors import (
    PreparedAnchor,
    prepare_anchors,
    validate_inputs,
)
from src.pipelines.guidance.slicegan.condition import (
    fit_noise,
    resolve_target_fraction,
    tune_condition,
)
from src.pipelines.guidance.slicegan.quality import (
    candidate_score,
    morphology_target,
    quality_passes,
    quality_score,
)
from src.pipelines.guidance.slicegan.volume import (
    calibrate,
    decode_probabilities,
    render_latent,
)
from src.pipelines.reconstruction.volume import decode_latents
from src.pipelines.training.slicegan import train_step


@dataclass(frozen=True)
class Candidate:
    step: int
    score: float
    generator: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]
    stats: dict[str, torch.Tensor]


def generate_conditional_slicegan(
    sampler,
    vae: torch.nn.Module,
    *,
    anchors: Sequence[VolumeAnchor],
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    volume_size: int,
    num_phases: int,
    config,
    device: torch.device,
    scale_batch_size: int = 16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    factor, latent_size = validate_inputs(
        vae,
        anchors=anchors,
        target_fraction=target_fraction,
        phase_fraction_tolerance=phase_fraction_tolerance,
        volume_size=volume_size,
        num_phases=num_phases,
    )
    train = config.train
    condition = config.condition
    prepared = prepare_anchors(
        vae,
        anchors,
        factor=factor,
        volume_size=volume_size,
        num_phases=num_phases,
        device=device,
    )
    target_fraction = resolve_target_fraction(
        prepared,
        target_fraction=target_fraction,
        num_phases=num_phases,
        device=device,
    )

    _sync(device)
    total_started = time.perf_counter()
    reference_started = time.perf_counter()
    anchor_latents, anchor_images = anchor_references(prepared)
    diffusion_latents, diffusion_images = diffusion_references(
        sampler,
        vae,
        count=train.reference_count,
        num_phases=num_phases,
    )
    total_steps = train.steps + train.mix_steps
    effective_mix = train.mix_probability * train.mix_steps / total_steps
    target = morphology_target(
        anchor_images,
        diffusion_images,
        mix_probability=effective_mix,
        target_fraction=target_fraction,
    )
    _sync(device)
    reference_seconds = time.perf_counter() - reference_started

    base_noise_size = int(vae.latent_size) // SCALE_FACTOR
    generator = SliceGANGenerator(int(vae.latent_ch)).to(device)
    critic = SliceGANCritic(int(vae.latent_ch)).to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=train.lr,
        betas=train.betas,
    )
    optimizer_d = torch.optim.Adam(
        critic.parameters(),
        lr=train.lr,
        betas=train.betas,
    )
    preview_noise = torch.randn(
        train.preview_count,
        NOISE_CHANNELS,
        base_noise_size,
        base_noise_size,
        base_noise_size,
        device=device,
    )
    events = {
        max(1, round(train.steps * fraction / 4))
        for fraction in range(1, 5)
    }
    events.update(
        train.steps + max(1, round(train.mix_steps * fraction / 2))
        for fraction in range(1, 3)
        if train.mix_steps > 0
    )
    candidates: list[Candidate] = []
    train_stats: dict[str, torch.Tensor] = {}
    training_started = time.perf_counter()
    for step in range(1, train.steps + train.mix_steps + 1):
        train_stats = train_step(
            generator,
            critic,
            optimizer_g,
            optimizer_d,
            anchor_latents,
            diffusion_latents,
            noise_size=base_noise_size,
            mixed=step > train.steps,
            config=train,
        )
        if step in events:
            score = candidate_score(
                generator,
                vae,
                preview_noise,
                target,
                num_phases=num_phases,
            )
            candidates.append(
                Candidate(
                    step=step,
                    score=float(score.item()),
                    generator={
                        name: value.detach().cpu().clone()
                        for name, value in generator.state_dict().items()
                    },
                    critic={
                        name: value.detach().cpu().clone()
                        for name, value in critic.state_dict().items()
                    },
                    stats={
                        name: value.detach().clone()
                        for name, value in train_stats.items()
                    },
                )
            )
            candidates.sort(key=lambda item: item.score)
            del candidates[max(condition.min_trials, 3) :]
    _sync(device)
    training_seconds = time.perf_counter() - training_started
    if not candidates:
        raise RuntimeError("SliceGAN training produced no candidate generator.")

    generation_started = time.perf_counter()
    selected_volume = None
    selected_score = torch.full((), float("inf"), device=device)
    selected_stats: dict[str, torch.Tensor] = {}
    selected_train_stats: dict[str, torch.Tensor] = {}
    selected_step = 0
    attempted = 0
    output_noise_size = latent_size // SCALE_FACTOR
    for candidate in candidates:
        if attempted >= condition.min_trials and quality_passes(
            selected_stats,
            condition=condition,
            phase_fraction_tolerance=phase_fraction_tolerance,
        ):
            break
        attempted += 1
        generator.load_state_dict(candidate.generator)
        critic.load_state_dict(candidate.critic)
        noise, noise_stats = fit_noise(
            generator,
            critic,
            vae,
            prepared,
            noise_size=output_noise_size,
            steps=condition.noise_steps,
            config=condition,
            device=device,
        )
        latent_before = render_latent(
            generator,
            noise,
            render=config.render,
        ).detach()
        latent, tune_stats = tune_condition(
            generator,
            critic,
            vae,
            noise,
            prepared,
            target_fraction,
            target_transition=target["transition"],
            factor=factor,
            steps=condition.tune_steps,
            config=condition,
            render=config.render,
            optimizer_betas=train.betas,
        )
        probabilities = decode_probabilities(
            vae,
            latent,
            num_phases=num_phases,
            batch_size=scale_batch_size,
        )
        volume = calibrate(
            probabilities,
            prepared,
            target_fraction=target_fraction,
            num_phases=num_phases,
        )
        quality, quality_stats = quality_score(
            volume,
            target,
            target_fraction=target_fraction,
            anchors=prepared,
            num_phases=num_phases,
            mismatch_tolerance=condition.mismatch_tolerance,
        )
        if quality < selected_score:
            selected_score = quality.detach()
            selected_volume = volume.detach().clone()
            selected_step = candidate.step
            selected_train_stats = candidate.stats
            selected_stats = {
                **noise_stats,
                **tune_stats,
                **quality_stats,
                "slicegan_changed_latent_fraction": (
                    (latent - latent_before).abs() > 1e-4
                ).float().mean(),
            }

    if selected_volume is None:
        raise RuntimeError("SliceGAN conditioning produced no candidate volume.")
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started
    total_seconds = time.perf_counter() - total_started

    phase_fraction = torch.stack(
        [
            (selected_volume == phase).float().mean()
            for phase in range(num_phases)
        ]
    )
    phase_error = (phase_fraction - target_fraction).abs()
    if float(phase_error.max().item()) > phase_fraction_tolerance:
        raise RuntimeError("generated phase fractions exceed phase_fraction_tolerance.")

    stats = {
        "slicegan_steps": torch.tensor(train.steps, device=device),
        "slicegan_mix_steps": torch.tensor(train.mix_steps, device=device),
        "slicegan_selected_step": torch.tensor(selected_step, device=device),
        "slicegan_candidates": torch.tensor(attempted, device=device),
        "slicegan_condition_quality": selected_score,
        "slicegan_latent_size": torch.tensor(latent_size, device=device),
        "slicegan_phase_fraction": phase_fraction,
        "slicegan_target_phase_fraction": target_fraction,
        "slicegan_phase_fraction_error": phase_error,
        "slicegan_phase_fraction_within_tolerance": (
            phase_error.max() <= phase_fraction_tolerance
        ),
        "slicegan_reference_seconds": torch.tensor(reference_seconds, device=device),
        "slicegan_training_seconds": torch.tensor(training_seconds, device=device),
        "slicegan_generation_seconds": torch.tensor(generation_seconds, device=device),
        "slicegan_total_seconds": torch.tensor(total_seconds, device=device),
        **selected_train_stats,
        **selected_stats,
    }
    return selected_volume.float(), stats


@torch.no_grad()
def diffusion_references(
    sampler,
    vae: torch.nn.Module,
    *,
    count: int,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vae.eval()
    latents = sampler.sample(
        (
            count,
            int(vae.latent_ch),
            int(vae.latent_size),
            int(vae.latent_size),
        )
    )
    _, probabilities = decode_latents(
        vae,
        latents,
        num_phases=num_phases,
    )
    labels = probabilities.argmax(dim=1)
    images = F.one_hot(labels, num_classes=num_phases).movedim(-1, 1).float()
    return latents.detach(), images


def anchor_references(
    anchors: Sequence[PreparedAnchor],
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_images = []
    categorical_images = []
    for anchor in anchors:
        for patch in anchor.patches:
            for flipped in (False, True):
                for turns in range(4):
                    latent = torch.rot90(patch.latent, turns, dims=(-2, -1))
                    probabilities = torch.rot90(
                        patch.probabilities,
                        turns,
                        dims=(-2, -1),
                    )
                    if flipped:
                        latent = torch.flip(latent, dims=(-1,))
                        probabilities = torch.flip(probabilities, dims=(-1,))
                    latent_images.append(latent)
                    categorical_images.append(probabilities)
    return torch.stack(latent_images), torch.stack(categorical_images)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
