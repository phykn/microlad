from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.app.api.options import QualityConfig


def check_quality(
    stats: dict[str, torch.Tensor],
    *,
    calibration_valid: bool,
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
    phase_fraction_tolerance: float,
    quality: "QualityConfig",
) -> tuple[bool, dict[str, torch.Tensor]]:
    zero = changed.new_zeros(())
    anchor = (
        torch.relu(stats["anchor_max_mismatch"] - quality.anchor_tolerance)
        if "anchor_max_mismatch" in stats
        else zero
    )
    fraction = (
        torch.relu(
            stats["phase_fraction_error"].abs().max()
            - phase_fraction_tolerance
        )
        if "phase_fraction_error" in stats
        else zero
    )
    if target_transition is None:
        rates = stats["axis_transition_rate"]
        transition = torch.relu(
            (rates.max() - rates.min()) - quality.morphology_tolerance
        )
    else:
        transition = torch.relu(
            (stats["axis_transition_rate"] - target_transition).abs().max()
            - quality.morphology_tolerance
        )
    run = (
        torch.relu(
            stats["axis_run_profile_mae"].max()
            - quality.morphology_tolerance
        )
        if "axis_run_profile_mae" in stats
        else zero
    )
    boundary = torch.relu(
        stats["axis_global_boundary_jump"].max()
        - quality.continuity_tolerance
    )
    repetition = torch.relu(
        stats["axis_exact_repeat_rate"].max() - quality.repeat_tolerance
    )
    budget = torch.relu(changed - quality.calibration_budget)
    calibration = zero if calibration_valid else changed.new_ones(())
    errors = {
        "anchor": anchor,
        "fraction": fraction,
        "transition": transition,
        "run": run,
        "boundary": boundary,
        "repetition": repetition,
        "calibration_budget": budget,
        "calibration": calibration,
    }
    passed = all(float(value.item()) == 0.0 for value in errors.values())
    return passed, errors


def quality_score(
    stats: dict[str, torch.Tensor],
    *,
    changed: torch.Tensor,
    target_transition: torch.Tensor | None,
) -> torch.Tensor:
    values = [changed]
    if "anchor_max_mismatch" in stats:
        values.append(stats["anchor_max_mismatch"])
    if "phase_fraction_error" in stats:
        values.append(stats["phase_fraction_error"].abs().max())
    values.extend(
        [
            stats["axis_exact_repeat_rate"].max(),
            stats["axis_global_boundary_jump"].max(),
            transition_error(stats["axis_transition_rate"], target_transition),
        ]
    )
    if "axis_run_profile_mae" in stats:
        values.append(stats["axis_run_profile_mae"].mean())
    return torch.stack(values).mean()


def reference_transition(references: torch.Tensor | None) -> torch.Tensor | None:
    if references is None:
        return None
    labels = references.argmax(dim=1)
    return 0.5 * (
        (labels[:, :, 1:] != labels[:, :, :-1]).float().mean()
        + (labels[:, 1:, :] != labels[:, :-1, :]).float().mean()
    )


def transition_error(
    rates: torch.Tensor,
    target: torch.Tensor | None,
) -> torch.Tensor:
    if target is None:
        return rates.max() - rates.min()
    return (rates - target).abs().mean()
