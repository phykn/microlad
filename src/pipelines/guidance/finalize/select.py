from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from src.modeling.phases import probabilities_to_calibrated_labels
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.finalize.quality import (
    check_quality,
    quality_score,
    reference_transition,
    transition_error,
)
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume
from src.pipelines.reconstruction.refinement import refine_probabilities
from src.pipelines.reconstruction.volume import decode_volume_probs

if TYPE_CHECKING:
    from src.app.api.options import QualityConfig, RefineConfig


@torch.no_grad()
def select_volume(
    vae: torch.nn.Module,
    latents: Sequence[torch.Tensor],
    *,
    candidate_steps: Sequence[int],
    num_phases: int,
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    anchors: Sequence[VolumeAnchor],
    references: torch.Tensor | None,
    refine: "RefineConfig",
    quality: "QualityConfig",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(latents) != len(candidate_steps):
        raise ValueError("latents and candidate_steps must have the same length.")
    if not latents:
        raise ValueError("at least one latent candidate is required.")

    anchor_mask = _anchor_mask(
        int(vae.image_size),
        anchors,
        device=latents[0].device,
    )
    target_transition = reference_transition(references)
    records = []
    for latent_step, latent in zip(candidate_steps, latents, strict=True):
        decoded = decode_volume_probs(vae, latent, num_phases=num_phases)
        base = decoded.argmax(dim=1)[0].float()
        for refine_steps in refine.candidates:
            probabilities = (
                decoded
                if refine_steps == 0
                else refine_probabilities(
                    base,
                    vae,
                    steps=refine_steps,
                    batch_size=refine.batch_size,
                    strength=refine.strength,
                    anchor_strength=refine.anchor_strength,
                    anchor_mask=anchor_mask,
                )
            )
            pre = probabilities.argmax(dim=1)[0].float()
            calibrated, calibration_valid = _calibrate(
                probabilities,
                target_fraction=target_fraction,
                num_phases=num_phases,
                anchor_mask=anchor_mask,
            )
            changed = (calibrated != pre).float().mean()
            pre_stats = evaluate_phase_volume(
                pre,
                num_phases=num_phases,
                references=references,
                target_fraction=target_fraction,
                anchors=anchors,
            )
            post_stats = evaluate_phase_volume(
                calibrated,
                num_phases=num_phases,
                references=references,
                target_fraction=target_fraction,
                anchors=anchors,
            )
            passed, errors = check_quality(
                post_stats,
                calibration_valid=calibration_valid,
                changed=changed,
                target_transition=target_transition,
                phase_fraction_tolerance=phase_fraction_tolerance,
                quality=quality,
            )
            score = quality_score(
                post_stats,
                changed=changed,
                target_transition=target_transition,
            )
            violation = torch.stack(tuple(errors.values())).sum()
            records.append(
                {
                    "volume": calibrated,
                    "latent_step": int(latent_step),
                    "refine_steps": int(refine_steps),
                    "passed": passed,
                    "score": score,
                    "violation": violation,
                    "changed": changed,
                    "pre": pre_stats,
                    "post": post_stats,
                    "errors": errors,
                }
            )

    feasible = [record for record in records if record["passed"]]
    if quality.strict and not feasible:
        best = min(
            records,
            key=lambda record: (
                float(record["violation"].item()),
                float(record["score"].item()),
            ),
        )
        failed = [
            name
            for name, value in best["errors"].items()
            if float(value.item()) > 0.0
        ]
        raise RuntimeError(
            "no prediction candidate passed the final quality gate: "
            + ", ".join(failed)
        )
    pool = feasible or records
    selected = min(
        pool,
        key=lambda record: (
            0.0 if record["passed"] else float(record["violation"].item()),
            *_selection_rank(
                record["post"],
                record["changed"],
                target_transition,
            ),
        ),
    )

    stats = {
        "candidate_count": torch.tensor(len(records), device=latents[0].device),
        "candidate_passes": torch.tensor(
            [bool(record["passed"]) for record in records],
            device=latents[0].device,
        ),
        "candidate_scores": torch.stack([record["score"] for record in records]),
        "candidate_violations": torch.stack(
            [record["violation"] for record in records]
        ),
        "selected_latent_step": torch.tensor(
            selected["latent_step"],
            device=latents[0].device,
        ),
        "selected_refine_steps": torch.tensor(
            selected["refine_steps"],
            device=latents[0].device,
        ),
        "quality_passed": torch.tensor(
            bool(selected["passed"]),
            device=latents[0].device,
        ),
        "calibration_changed_fraction": selected["changed"],
    }
    stats.update(
        {f"pre_calibration_{name}": value for name, value in selected["pre"].items()}
    )
    stats.update(
        {f"final_{name}": value for name, value in selected["post"].items()}
    )
    stats.update(
        {f"quality_{name}": value for name, value in selected["errors"].items()}
    )
    if anchors:
        stats["calibration_anchor_delta"] = (
            selected["post"]["anchor_mismatches"]
            - selected["pre"]["anchor_mismatches"]
        )
    stats["calibration_transition_delta"] = (
        selected["post"]["axis_transition_rate"]
        - selected["pre"]["axis_transition_rate"]
    )
    stats["calibration_boundary_delta"] = (
        selected["post"]["axis_global_boundary_jump"]
        - selected["pre"]["axis_global_boundary_jump"]
    )
    if references is not None:
        stats["calibration_run_delta"] = (
            selected["post"]["axis_run_profile_mae"]
            - selected["pre"]["axis_run_profile_mae"]
        )
    return selected["volume"].float(), stats


def _calibrate(
    probabilities: torch.Tensor,
    *,
    target_fraction: torch.Tensor | None,
    num_phases: int,
    anchor_mask: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    selected = probabilities.argmax(dim=1, keepdim=True)
    if target_fraction is None:
        return selected[0, 0].float(), True
    try:
        labels = probabilities_to_calibrated_labels(
            probabilities,
            num_phases,
            target_fractions=target_fraction,
            fixed_labels=selected,
            fixed_mask=anchor_mask,
        )
    except (RuntimeError, ValueError):
        return selected[0, 0].float(), False
    return labels[0, 0].float(), True


def _anchor_mask(
    size: int,
    anchors: Sequence[VolumeAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size, size, dtype=torch.bool, device=device)
    for anchor in anchors:
        length = int(anchor.image.shape[-1])
        start = int(anchor.start)
        stop = start + length
        if anchor.axis == 0:
            mask[:, :, anchor.index, start:stop, start:stop] = True
        elif anchor.axis == 1:
            mask[:, :, start:stop, anchor.index, start:stop] = True
        else:
            mask[:, :, start:stop, start:stop, anchor.index] = True
    return mask


def _selection_rank(
    stats: dict[str, torch.Tensor],
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
) -> tuple[float, ...]:
    zero = changed.new_zeros(())
    return tuple(
        float(value.item())
        for value in (
            stats.get("anchor_max_mismatch", zero),
            stats.get("phase_fraction_error", zero).abs().max(),
            changed,
            stats["axis_exact_repeat_rate"].max(),
            stats["axis_global_boundary_jump"].max(),
            transition_error(stats["axis_transition_rate"], target_transition),
            stats.get("axis_run_profile_mae", zero).mean(),
        )
    )
