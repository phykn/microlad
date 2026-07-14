from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from src.modeling.phases import probabilities_to_calibrated_labels
from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.conditioning.prepare import build_volume_anchor_mask
from src.pipelines.finalize.quality import (
    check_quality,
    quality_score,
    reference_transition,
    transition_error,
)
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume
from src.pipelines.reconstruction.refine import refine_probabilities
from src.pipelines.reconstruction.volume import decode_volume_probs

if TYPE_CHECKING:
    from src.app.api.options import QualityConfig, RefineConfig


@torch.no_grad()
def select_latent_volume(
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

    size = int(vae.image_size)
    anchor_mask = build_volume_anchor_mask(
        (size, size, size),
        anchors,
        device=latents[0].device,
    )
    target_transition = reference_transition(references)
    base_std = latents[0].std().clamp_min(1e-6)
    records = []
    for latent_step, latent in zip(candidate_steps, latents, strict=True):
        latent_delta = (latent - latents[0]).square().mean().sqrt() / base_std
        decoded = decode_volume_probs(vae, latent, num_phases=num_phases)
        decoded_stats = evaluate_phase_volume(
            decoded.argmax(dim=1)[0].float(),
            num_phases=num_phases,
            references=references,
            target_fraction=target_fraction,
            anchors=anchors,
        )
        for refine_steps in refine.candidates:
            probabilities = (
                decoded
                if refine_steps == 0
                else refine_probabilities(
                    decoded,
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
                    "latent_delta": latent_delta,
                    "decoded": decoded_stats,
                    "pre": pre_stats,
                    "post": post_stats,
                    "errors": errors,
                }
            )

    feasible = [record for record in records if record["passed"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda record: _morphology_rank(
                record["post"],
                record["changed"],
                target_transition,
                record["latent_delta"],
            ),
        )
    else:
        selected = min(
            records,
            key=lambda record: _violation_rank(
                record["errors"],
                record["post"],
                record["changed"],
                target_transition,
                record["latent_delta"],
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
        "candidate_latent_steps": torch.tensor(
            [record["latent_step"] for record in records],
            device=latents[0].device,
        ),
        "candidate_refine_steps": torch.tensor(
            [record["refine_steps"] for record in records],
            device=latents[0].device,
        ),
        "candidate_calibration_changed_fractions": torch.stack(
            [record["changed"] for record in records]
        ),
        "candidate_latent_delta_over_base_std": torch.stack(
            [record["latent_delta"] for record in records]
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
        "selected_latent_delta_over_base_std": selected["latent_delta"],
    }
    stats.update(
        {f"pre_calibration_{name}": value for name, value in selected["pre"].items()}
    )
    stats.update({f"final_{name}": value for name, value in selected["post"].items()})
    stats.update(
        {f"quality_{name}": value for name, value in selected["errors"].items()}
    )
    for stage, key in (
        ("decoded", "decoded"),
        ("refined", "pre"),
        ("final", "post"),
    ):
        for metric in (
            "phase_fraction",
            "axis_transition_rate",
            "axis_exact_repeat_rate",
            "axis_near_repeat_rate",
            "axis_max_repeat_streak",
            "axis_global_boundary_jump",
            "axis_run_profile_mae",
            "component_count",
            "euler_3d_density",
            "phase_axis_percolation",
        ):
            if metric in records[0][key]:
                stats[f"candidate_{stage}_{metric}"] = torch.stack(
                    [record[key][metric] for record in records]
                )
    if anchors:
        stats["candidate_decoded_anchor_mismatches"] = torch.stack(
            [record["decoded"]["anchor_mismatches"] for record in records]
        )
        stats["candidate_refined_anchor_mismatches"] = torch.stack(
            [record["pre"]["anchor_mismatches"] for record in records]
        )
        stats["candidate_final_anchor_mismatches"] = torch.stack(
            [record["post"]["anchor_mismatches"] for record in records]
        )
        stats["calibration_anchor_delta"] = (
            selected["post"]["anchor_mismatches"] - selected["pre"]["anchor_mismatches"]
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


@torch.no_grad()
def select_probability_volume(
    probabilities: Sequence[torch.Tensor],
    *,
    candidate_steps: Sequence[int],
    refine_steps: Sequence[int],
    num_phases: int,
    target_fraction: torch.Tensor | None,
    phase_fraction_tolerance: float,
    anchors: Sequence[VolumeAnchor],
    references: torch.Tensor | None,
    quality: "QualityConfig",
    candidate_deltas: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    count = len(probabilities)
    if count == 0 or len(candidate_steps) != count or len(refine_steps) != count:
        raise ValueError(
            "probabilities, candidate_steps, and refine_steps must have the same "
            "non-zero length."
        )
    if candidate_deltas is not None and len(candidate_deltas) != count:
        raise ValueError("candidate_deltas must match probabilities length.")
    first = probabilities[0]
    if first.ndim != 5 or first.shape[:2] != (1, num_phases):
        raise ValueError("probabilities must have shape [1, P, D, H, W].")
    if len(set(map(int, first.shape[2:]))) != 1:
        raise ValueError("probability volumes must be cubic.")
    if any(value.shape != first.shape for value in probabilities):
        raise ValueError("probability candidates must have the same shape.")

    size = int(first.shape[2])
    anchor_mask = build_volume_anchor_mask(
        (size, size, size),
        anchors,
        device=first.device,
    )
    target_transition = reference_transition(references)
    records = []
    for index, (step, refine, candidate) in enumerate(
        zip(candidate_steps, refine_steps, probabilities, strict=True)
    ):
        if not candidate.is_floating_point() or not torch.isfinite(candidate).all():
            raise ValueError("probability candidates must be finite floating point.")
        if torch.any(candidate < 0.0) or torch.any(candidate.sum(dim=1) <= 0.0):
            raise ValueError("probability candidates must contain positive phase mass.")
        normalized = candidate / candidate.sum(dim=1, keepdim=True)
        pre = normalized.argmax(dim=1)[0].float()
        calibrated, calibration_valid = _calibrate(
            normalized,
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
        delta = (
            changed.new_zeros(())
            if candidate_deltas is None
            else candidate_deltas[index].to(device=changed.device)
        )
        records.append(
            {
                "volume": calibrated,
                "step": int(step),
                "refine": int(refine),
                "passed": passed,
                "score": quality_score(
                    post_stats,
                    changed=changed,
                    target_transition=target_transition,
                ),
                "violation": torch.stack(tuple(errors.values())).sum(),
                "changed": changed,
                "delta": delta,
                "pre": pre_stats,
                "post": post_stats,
                "errors": errors,
            }
        )
    feasible = [record for record in records if record["passed"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda record: _morphology_rank(
                record["post"],
                record["changed"],
                target_transition,
                record["delta"],
            ),
        )
    else:
        selected = min(
            records,
            key=lambda record: _violation_rank(
                record["errors"],
                record["post"],
                record["changed"],
                target_transition,
                record["delta"],
            ),
        )
    device = first.device
    result = {
        "candidate_count": torch.tensor(len(records), device=device),
        "candidate_passes": torch.tensor(
            [bool(record["passed"]) for record in records],
            device=device,
        ),
        "candidate_scores": torch.stack([record["score"] for record in records]),
        "candidate_violations": torch.stack(
            [record["violation"] for record in records]
        ),
        "candidate_scale_steps": torch.tensor(
            [record["step"] for record in records],
            device=device,
        ),
        "candidate_refine_steps": torch.tensor(
            [record["refine"] for record in records],
            device=device,
        ),
        "candidate_calibration_changed_fractions": torch.stack(
            [record["changed"] for record in records]
        ),
        "candidate_latent_delta_over_base_std": torch.stack(
            [record["delta"] for record in records]
        ),
        "selected_scale_step": torch.tensor(selected["step"], device=device),
        "selected_refine_steps": torch.tensor(selected["refine"], device=device),
        "quality_passed": torch.tensor(bool(selected["passed"]), device=device),
        "calibration_changed_fraction": selected["changed"],
        "selected_latent_delta_over_base_std": selected["delta"],
    }
    result.update(
        {f"pre_calibration_{name}": value for name, value in selected["pre"].items()}
    )
    result.update({f"final_{name}": value for name, value in selected["post"].items()})
    result.update(
        {f"quality_{name}": value for name, value in selected["errors"].items()}
    )
    for stage, key in (("refined", "pre"), ("final", "post")):
        for metric in (
            "phase_fraction",
            "axis_transition_rate",
            "axis_exact_repeat_rate",
            "axis_near_repeat_rate",
            "axis_max_repeat_streak",
            "axis_global_boundary_jump",
            "axis_run_profile_mae",
            "component_count",
            "euler_3d_density",
            "phase_axis_percolation",
        ):
            if metric in records[0][key]:
                result[f"candidate_{stage}_{metric}"] = torch.stack(
                    [record[key][metric] for record in records]
                )
    if anchors:
        result["candidate_refined_anchor_mismatches"] = torch.stack(
            [record["pre"]["anchor_mismatches"] for record in records]
        )
        result["candidate_final_anchor_mismatches"] = torch.stack(
            [record["post"]["anchor_mismatches"] for record in records]
        )
        result["calibration_anchor_delta"] = (
            selected["post"]["anchor_mismatches"]
            - selected["pre"]["anchor_mismatches"]
        )
    result["calibration_transition_delta"] = (
        selected["post"]["axis_transition_rate"]
        - selected["pre"]["axis_transition_rate"]
    )
    result["calibration_boundary_delta"] = (
        selected["post"]["axis_global_boundary_jump"]
        - selected["pre"]["axis_global_boundary_jump"]
    )
    if references is not None:
        result["calibration_run_delta"] = (
            selected["post"]["axis_run_profile_mae"]
            - selected["pre"]["axis_run_profile_mae"]
        )
    return selected["volume"], result


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


def _selection_rank(
    stats: dict[str, torch.Tensor],
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
    latent_delta: torch.Tensor | None = None,
) -> tuple[float, ...]:
    zero = changed.new_zeros(())
    delta = zero if latent_delta is None else latent_delta
    return tuple(
        float(value.item())
        for value in (
            stats.get("anchor_max_mismatch", zero),
            stats.get("anchor_max_phase_mismatch", zero),
            stats.get("phase_fraction_error", zero).abs().max(),
            changed,
            stats["axis_exact_repeat_rate"].max(),
            stats.get("axis_near_repeat_rate", zero).max(),
            stats["axis_global_boundary_jump"].max(),
            transition_error(stats["axis_transition_rate"], target_transition),
            stats.get("axis_run_profile_mae", zero).mean(),
            delta,
        )
    )


def _morphology_rank(
    stats: dict[str, torch.Tensor],
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
    latent_delta: torch.Tensor | None = None,
) -> tuple[float, ...]:
    zero = changed.new_zeros(())
    delta = zero if latent_delta is None else latent_delta
    return tuple(
        float(value.item())
        for value in (
            stats["axis_exact_repeat_rate"].max(),
            stats.get("axis_near_repeat_rate", zero).max(),
            stats.get("axis_max_repeat_streak", zero).max(),
            stats["axis_global_boundary_jump"].max(),
            stats.get("axis_run_profile_mae", zero).mean(),
            transition_error(stats["axis_transition_rate"], target_transition),
            changed,
            stats.get("anchor_max_mismatch", zero),
            stats.get("phase_fraction_error", zero).abs().max(),
            delta,
        )
    )


def _violation_rank(
    errors: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
    latent_delta: torch.Tensor | None = None,
) -> tuple[float, ...]:
    return tuple(
        float(value.item())
        for value in (
            errors["anchor"],
            stats.get("anchor_max_phase_mismatch", changed.new_zeros(())),
            errors["fraction"],
            errors["calibration"],
            errors["calibration_budget"],
            errors["repetition"],
            errors["boundary"],
            errors["transition"],
            errors["run"],
        )
    ) + _selection_rank(stats, changed, target_transition, latent_delta)
