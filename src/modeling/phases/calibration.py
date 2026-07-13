import torch


def _validate_num_phases(num_phases: int) -> None:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")


@torch.no_grad()
def probabilities_to_calibrated_labels(
    probabilities: torch.Tensor,
    num_phases: int,
    target_fractions: torch.Tensor | None = None,
    fixed_labels: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if probabilities.ndim < 3:
        raise ValueError(
            "probabilities must have shape [B, num_phases, ...spatial]."
        )
    if probabilities.shape[1] != num_phases:
        raise ValueError("probability channels must match num_phases.")
    if any(size <= 0 for size in probabilities.shape):
        raise ValueError("probabilities must not be empty.")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite.")
    if torch.any(probabilities < 0.0) or torch.any(probabilities > 1.0):
        raise ValueError("probabilities must be between 0 and 1.")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("probabilities must sum to one across phases.")

    spatial_shape = probabilities.shape[2:]
    fractions = _validate_target_fractions(
        target_fractions,
        num_phases=num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    fixed_labels, fixed_mask = _validate_fixed_labels(
        fixed_labels,
        fixed_mask,
        probabilities=probabilities,
        num_phases=num_phases,
    )
    calibrated = []
    for sample_index, sample in enumerate(probabilities):
        flat = sample.reshape(num_phases, -1)
        if fixed_mask is None or fixed_labels is None:
            labels = _calibrate_sample_labels(flat, target_fractions=fractions)
        else:
            sample_mask = fixed_mask[sample_index].reshape(-1)
            sample_fixed = fixed_labels[sample_index].reshape(-1)
            labels = _calibrate_with_fixed_labels(
                flat,
                sample_fixed,
                sample_mask,
                target_fractions=fractions,
            )
        calibrated.append(labels.reshape(1, *spatial_shape))

    return torch.stack(calibrated, dim=0)


def _validate_target_fractions(
    target_fractions: torch.Tensor | None,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if target_fractions is None:
        return None
    fractions = torch.as_tensor(target_fractions, device=device, dtype=dtype)
    if fractions.shape != (num_phases,):
        raise ValueError("target_fractions must have shape [num_phases].")
    if not torch.isfinite(fractions).all():
        raise ValueError("target_fractions must be finite.")
    if torch.any(fractions < 0.0):
        raise ValueError("target_fractions must be non-negative.")
    if not torch.allclose(
        fractions.sum(),
        torch.ones((), device=device, dtype=dtype),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("target_fractions must sum to one.")
    return fractions


def _validate_fixed_labels(
    fixed_labels: torch.Tensor | None,
    fixed_mask: torch.Tensor | None,
    *,
    probabilities: torch.Tensor,
    num_phases: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if fixed_labels is None and fixed_mask is None:
        return None, None
    if fixed_labels is None or fixed_mask is None:
        raise ValueError("fixed_labels and fixed_mask must be provided together.")

    expected_shape = (probabilities.shape[0], 1, *probabilities.shape[2:])
    labels = torch.as_tensor(fixed_labels, device=probabilities.device)
    mask = torch.as_tensor(fixed_mask, device=probabilities.device)
    if labels.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(
            "fixed_labels and fixed_mask must have shape [B, 1, ...spatial]."
        )
    if not torch.isfinite(labels).all():
        raise ValueError("fixed_labels must be finite.")
    if not torch.equal(labels, labels.round()):
        raise ValueError("fixed_labels must contain integer phase labels.")
    if labels.min().item() < 0 or labels.max().item() >= num_phases:
        raise ValueError("fixed_labels must be inside the phase range.")
    return labels.to(torch.long), mask.to(torch.bool)


def _calibrate_with_fixed_labels(
    probabilities: torch.Tensor,
    fixed_labels: torch.Tensor,
    fixed_mask: torch.Tensor,
    *,
    target_fractions: torch.Tensor | None,
) -> torch.Tensor:
    num_phases, num_pixels = probabilities.shape
    mass = (
        probabilities.sum(dim=1)
        if target_fractions is None
        else target_fractions * num_pixels
    )
    target_counts = _counts_from_mass(mass, num_pixels)
    fixed_counts = torch.bincount(
        fixed_labels[fixed_mask],
        minlength=num_phases,
    )
    if target_fractions is None:
        target_counts = _reserve_fixed_counts(target_counts, fixed_counts)
    free_target_counts = target_counts - fixed_counts
    if torch.any(free_target_counts < 0):
        raise ValueError("fixed labels exceed the requested phase counts.")

    labels = fixed_labels.clone()
    free_mask = ~fixed_mask
    if bool(free_mask.any().item()):
        labels[free_mask] = _calibrate_sample_labels(
            probabilities[:, free_mask],
            target_counts=free_target_counts,
        )
    elif torch.any(free_target_counts != 0):
        raise ValueError("fixed labels do not match the requested phase counts.")
    return labels


def _reserve_fixed_counts(
    target_counts: torch.Tensor,
    fixed_counts: torch.Tensor,
) -> torch.Tensor:
    adjusted = torch.maximum(target_counts, fixed_counts)
    excess = int((adjusted.sum() - target_counts.sum()).item())
    while excess > 0:
        surplus = adjusted - fixed_counts
        phase = int(surplus.argmax().item())
        available = int(surplus[phase].item())
        if available <= 0:
            raise ValueError("fixed labels leave no free categorical capacity.")
        take = min(excess, available)
        adjusted[phase] -= take
        excess -= take
    return adjusted


def _calibrate_sample_labels(
    probabilities: torch.Tensor,
    *,
    target_fractions: torch.Tensor | None = None,
    target_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    num_phases, num_pixels = probabilities.shape
    if target_counts is None:
        mass = (
            probabilities.sum(dim=1)
            if target_fractions is None
            else target_fractions * num_pixels
        )
        target_counts = _counts_from_mass(mass, num_pixels)
    else:
        target_counts = target_counts.to(
            device=probabilities.device,
            dtype=torch.long,
        )
        if target_counts.shape != (num_phases,) or torch.any(target_counts < 0):
            raise ValueError("target_counts must contain one non-negative count per phase.")
        if int(target_counts.sum().item()) != num_pixels:
            raise ValueError("target_counts must sum to the number of pixels.")

    labels = probabilities.argmax(dim=0)
    counts = torch.bincount(labels, minlength=num_phases)
    tiny = torch.finfo(probabilities.dtype).tiny
    log_probabilities = probabilities.clamp_min(tiny).log()

    for target_phase in range(num_phases):
        need = int((target_counts[target_phase] - counts[target_phase]).item())
        if need <= 0:
            continue

        candidate_indices = []
        candidate_costs = []
        for source_phase in range(num_phases):
            surplus = int((counts[source_phase] - target_counts[source_phase]).item())
            if source_phase == target_phase or surplus <= 0:
                continue

            source_indices = torch.nonzero(labels == source_phase, as_tuple=False).flatten()
            costs = (
                log_probabilities[source_phase, source_indices]
                - log_probabilities[target_phase, source_indices]
            )
            take = min(surplus, int(source_indices.numel()))
            selected = costs.topk(take, largest=False).indices
            candidate_indices.append(source_indices[selected])
            candidate_costs.append(costs[selected])

        if not candidate_indices:
            raise RuntimeError("unable to calibrate categorical phase counts.")

        indices = torch.cat(candidate_indices)
        costs = torch.cat(candidate_costs)
        selected = costs.topk(need, largest=False).indices
        chosen = indices[selected]
        source_counts = torch.bincount(labels[chosen], minlength=num_phases)
        labels[chosen] = target_phase
        counts -= source_counts
        counts[target_phase] += chosen.numel()

    return labels


def _counts_from_mass(mass: torch.Tensor, num_pixels: int) -> torch.Tensor:
    target_counts = mass.floor().to(torch.long)
    remainder = num_pixels - int(target_counts.sum().item())
    if remainder > 0:
        fractions = mass - target_counts.to(mass.dtype)
        target_counts[fractions.topk(remainder).indices] += 1
    return target_counts
