import math

import pytest
import torch

from src.train.anchor import sample_anchor_condition


def test_sample_anchor_condition_returns_clean_and_boolean_mask() -> None:
    clean = torch.randn(32, 3, 8, 11)

    anchor, mask = sample_anchor_condition(
        clean,
        generator=torch.Generator().manual_seed(17),
    )

    assert anchor is clean
    assert mask.shape == (32, 1, 8, 11)
    assert mask.dtype == torch.bool
    assert mask.device == clean.device


def test_sample_anchor_condition_is_reproducible() -> None:
    clean = torch.randn(64, 2, 5, 7)
    first_generator = torch.Generator().manual_seed(1234)
    second_generator = torch.Generator().manual_seed(1234)

    _, first = sample_anchor_condition(clean, generator=first_generator)
    _, second = sample_anchor_condition(clean, generator=second_generator)

    assert torch.equal(first, second)


def test_sample_anchor_condition_covers_training_mask_shapes() -> None:
    height, width = 8, 11
    clean = torch.zeros(4096, 2, height, width)

    _, masks = sample_anchor_condition(
        clean,
        generator=torch.Generator().manual_seed(9),
    )
    masks = masks[:, 0]
    coverage = masks.sum(dim=(1, 2))

    assert (coverage == 0).any()
    assert (coverage == height * width).any()

    occupied_rows = masks.any(dim=2).sum(dim=1)
    occupied_columns = masks.any(dim=1).sum(dim=1)
    has_single_full_line = ((occupied_rows == 1) & (coverage == width)) | (
        (occupied_columns == 1) & (coverage == height)
    )
    assert has_single_full_line.any()

    has_partial_segment = (
        (occupied_rows == 1) & (coverage < width) & (coverage > 0)
    ) | ((occupied_columns == 1) & (coverage < height) & (coverage > 0))
    assert has_partial_segment.any()

    rectangular = []
    for mask in masks:
        positions = mask.nonzero()
        if positions.numel() == 0:
            rectangular.append(False)
            continue
        top, left = positions.amin(dim=0)
        bottom, right = positions.amax(dim=0)
        box = mask[top : bottom + 1, left : right + 1]
        rectangular.append(
            bool(box.all())
            and bottom > top
            and right > left
            and int(mask.sum()) < height * width
        )
    assert any(rectangular)

    has_cross = (
        (masks.all(dim=2).sum(dim=1) == 1)
        & (masks.all(dim=1).sum(dim=1) == 1)
        & (coverage == height + width - 1)
    )
    assert has_cross.any()

    has_parallel_lines = (
        (masks.all(dim=2).sum(dim=1) == 2) & (coverage == 2 * width)
    ) | ((masks.all(dim=1).sum(dim=1) == 2) & (coverage == 2 * height))
    assert has_parallel_lines.any()


def test_sample_anchor_condition_uses_configured_empty_probability() -> None:
    clean = torch.zeros(2048, 1, 4, 6)

    _, no_empty = sample_anchor_condition(
        clean,
        empty_probability=0.0,
        generator=torch.Generator().manual_seed(3),
    )
    _, all_empty = sample_anchor_condition(
        clean,
        empty_probability=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    _, default = sample_anchor_condition(
        clean,
        generator=torch.Generator().manual_seed(3),
    )

    assert no_empty.any(dim=(1, 2, 3)).all()
    assert not all_empty.any()
    empty_fraction = (~default.any(dim=(1, 2, 3))).float().mean()
    assert empty_fraction == pytest.approx(0.2, abs=0.025)


@pytest.mark.parametrize(
    ("clean", "message"),
    [
        (torch.zeros(2, 3, 4), "shape"),
        (torch.zeros(0, 1, 4, 4), "batch"),
        (torch.zeros(1, 0, 4, 4), "phase"),
        (torch.zeros(1, 1, 1, 4), "at least 2"),
        (torch.zeros(1, 1, 4, 1), "at least 2"),
    ],
)
def test_sample_anchor_condition_validates_clean(
    clean: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_anchor_condition(clean)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, math.inf, math.nan, True, "0.2"],
)
def test_sample_anchor_condition_validates_empty_probability(
    probability: object,
) -> None:
    with pytest.raises(ValueError, match="empty_probability"):
        sample_anchor_condition(
            torch.zeros(1, 1, 4, 4),
            empty_probability=probability,  # type: ignore[arg-type]
        )


def test_sample_anchor_condition_validates_argument_types() -> None:
    with pytest.raises(TypeError, match="clean"):
        sample_anchor_condition([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="generator"):
        sample_anchor_condition(
            torch.zeros(1, 1, 4, 4),
            generator=object(),  # type: ignore[arg-type]
        )
