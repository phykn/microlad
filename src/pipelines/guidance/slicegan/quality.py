from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.modeling.slicegan import SliceGANGenerator
from src.pipelines.guidance.metrics.runs import compute_run_profile
from src.pipelines.guidance.slicegan.anchors import (
    PreparedAnchor,
    anchor_mismatches,
    local_boundary_stats,
)

if TYPE_CHECKING:
    from src.app.api.options import SliceGANConditionConfig


RUN_LENGTHS = (2, 4, 8, 16)


def morphology_target(
    anchors: torch.Tensor,
    diffusion: torch.Tensor,
    *,
    mix_probability: float,
    target_fraction: torch.Tensor,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    anchor = reference_stats(anchors)
    diff = reference_stats(diffusion)
    return {
        "phase_fraction": target_fraction,
        "transition": (
            (1.0 - mix_probability) * anchor["transition"]
            + mix_probability * diff["transition"]
        ),
        "run_profile": (
            (1.0 - mix_probability) * anchor["run_profile"]
            + mix_probability * diff["run_profile"]
        ),
        "run_lengths": anchor["run_lengths"],
    }


def reference_stats(
    references: torch.Tensor,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    labels = references.argmax(dim=1)
    transition = 0.5 * (
        (labels[:, :, 1:] != labels[:, :, :-1]).float().mean()
        + (labels[:, 1:, :] != labels[:, :-1, :]).float().mean()
    )
    lengths = tuple(
        length for length in RUN_LENGTHS if length <= min(references.shape[-2:])
    )
    return {
        "transition": transition,
        "run_profile": compute_run_profile(references, lengths=lengths).mean(dim=0),
        "run_lengths": lengths,
    }


@torch.no_grad()
def candidate_score(
    generator: SliceGANGenerator,
    vae: torch.nn.Module,
    noises: torch.Tensor,
    target: dict[str, torch.Tensor | tuple[int, ...]],
    *,
    num_phases: int,
) -> torch.Tensor:
    generator.eval()
    volumes = generator(noises)
    scores = []
    for volume in volumes:
        references = _decode_references(vae, volume)
        scores.append(reference_error(references, target, num_phases=num_phases))
    generator.train()
    scores = torch.stack(scores)
    return scores.mean() + 0.5 * scores.max()


def reference_error(
    references: torch.Tensor,
    target: dict[str, torch.Tensor | tuple[int, ...]],
    *,
    num_phases: int,
) -> torch.Tensor:
    labels = references.argmax(dim=1)
    phase = F.one_hot(labels, num_classes=num_phases).float().mean(dim=(0, 1, 2))
    phase_error = (phase - target["phase_fraction"]).abs().mean()
    transition = 0.5 * (
        (labels[:, :, 1:] != labels[:, :, :-1]).float().mean()
        + (labels[:, 1:, :] != labels[:, :-1, :]).float().mean()
    )
    run = compute_run_profile(
        F.one_hot(labels, num_classes=num_phases).movedim(-1, 1).float(),
        lengths=target["run_lengths"],
    ).mean(dim=0)
    return (
        phase_error
        + (transition - target["transition"]).abs()
        + (run - target["run_profile"]).abs().mean()
    )


def quality_score(
    volume: torch.Tensor,
    target: dict[str, torch.Tensor | tuple[int, ...]],
    *,
    target_fraction: torch.Tensor,
    anchors: list[PreparedAnchor],
    num_phases: int,
    mismatch_tolerance: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = F.one_hot(
        volume.long(),
        num_classes=num_phases,
    ).movedim(-1, 0).unsqueeze(0).float()
    phase = probabilities.mean(dim=(0, 2, 3, 4))
    phase_error = (phase - target_fraction).abs().mean()
    transitions = torch.stack(
        [
            (
                volume.narrow(axis, 1, volume.shape[axis] - 1)
                != volume.narrow(axis, 0, volume.shape[axis] - 1)
            ).float().mean()
            for axis in range(3)
        ]
    )
    transition_error = (transitions - target["transition"]).abs().mean()
    run = compute_run_profile(probabilities, lengths=target["run_lengths"])
    run_error = (run - target["run_profile"].unsqueeze(0)).abs().mean()
    mismatches = anchor_mismatches(volume, anchors)
    boundary = [local_boundary_stats(volume, anchor) for anchor in anchors]
    boundary_std = torch.stack([value[0] for value in boundary]).max()
    boundary_jump = torch.stack([value[1] for value in boundary]).max()
    quality = (
        phase_error
        + transition_error
        + run_error
        + 0.2 * mismatches.mean()
        + boundary_std
        + boundary_jump
        + 10.0 * F.relu(mismatches.max() - mismatch_tolerance)
    )
    return quality, {
        "slicegan_quality_anchor_mismatch": mismatches.mean(),
        "slicegan_quality_anchor_mismatches": mismatches,
        "slicegan_quality_anchor_max_mismatch": mismatches.max(),
        "slicegan_quality_phase_mae": phase_error,
        "slicegan_quality_transition_mae": transition_error,
        "slicegan_quality_run_mae": run_error,
        "slicegan_quality_boundary_std": boundary_std,
        "slicegan_quality_boundary_jump": boundary_jump,
    }


def quality_passes(
    stats: dict[str, torch.Tensor],
    *,
    condition: "SliceGANConditionConfig",
    phase_fraction_tolerance: float,
) -> bool:
    if not stats:
        return False
    return (
        float(stats["slicegan_quality_anchor_max_mismatch"].item())
        <= condition.mismatch_tolerance
        and float(stats["slicegan_quality_phase_mae"].item())
        <= phase_fraction_tolerance
        and float(stats["slicegan_quality_transition_mae"].item())
        <= condition.morphology_tolerance
        and float(stats["slicegan_quality_run_mae"].item())
        <= condition.morphology_tolerance
        and float(stats["slicegan_quality_boundary_std"].item())
        <= condition.continuity_tolerance
        and float(stats["slicegan_quality_boundary_jump"].item())
        <= condition.continuity_tolerance
    )


@torch.no_grad()
def _decode_references(
    vae: torch.nn.Module,
    volume: torch.Tensor,
) -> torch.Tensor:
    latent_size = int(vae.latent_size)
    batches = []
    for axis in range(3):
        length = int(volume.shape[axis + 1])
        indices = torch.linspace(
            0,
            length - 1,
            steps=min(length, 8),
            device=volume.device,
        ).round().long()
        for index in indices.tolist():
            plane = volume.select(axis + 1, int(index))
            row = max((int(plane.shape[-2]) - latent_size) // 2, 0)
            col = max((int(plane.shape[-1]) - latent_size) // 2, 0)
            batches.append(
                plane[:, row : row + latent_size, col : col + latent_size]
            )
    return vae.decode_probs(torch.stack(batches))
