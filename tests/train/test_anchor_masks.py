import pytest
import torch

from src.train.anchor import _merge_multiple, sample_anchor_condition


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

    occupied_rows = masks.any(dim=2).sum(dim=1)
    occupied_columns = masks.any(dim=1).sum(dim=1)
    has_segment = ((occupied_rows == 1) | (occupied_columns == 1)) & (coverage > 0)
    assert has_segment.any()

    crops = []
    compounds = []
    for mask in masks:
        positions = mask.nonzero()
        if positions.numel() == 0:
            crops.append(False)
            compounds.append(False)
            continue
        top, left = positions.amin(dim=0)
        bottom, right = positions.amax(dim=0)
        box = mask[top : bottom + 1, left : right + 1]
        square = bool(box.all()) and bottom - top == right - left
        line = bottom == top or right == left
        crops.append(square and not line)
        compounds.append(not square and not line)
    assert any(crops)
    assert any(compounds)


def test_sample_anchor_condition_uses_equal_category_probability() -> None:
    clean = torch.zeros(8192, 1, 8, 8)

    _, masks = sample_anchor_condition(
        clean,
        generator=torch.Generator().manual_seed(3),
    )

    empty = (~masks.any(dim=(1, 2, 3))).float().mean()
    assert empty == pytest.approx(0.25, abs=0.02)

    area = masks[:, 0].sum(dim=(1, 2))
    lo = 2
    filled_squares = []
    for mask in masks[:, 0]:
        pos = mask.nonzero()
        if pos.numel() == 0:
            filled_squares.append(False)
            continue
        top, left = pos.amin(dim=0)
        bottom, right = pos.amax(dim=0)
        box = mask[top : bottom + 1, left : right + 1]
        filled_squares.append(
            bool(box.all())
            and bottom - top == right - left
            and lo <= int(bottom - top + 1) <= 8
        )
    assert any(filled_squares)
    assert int(area.max()) > lo * lo


def test_multiple_mask_falls_back_when_one_constraint_contains_another() -> None:
    a = torch.ones(2, 4, 4, dtype=torch.bool)
    b = torch.zeros_like(a)
    b[:, 1, 1] = True
    fallback = torch.zeros_like(a)
    fallback[:, 0] = True
    fallback[:, :, 0] = True

    mask = _merge_multiple(a, b, fallback)

    assert torch.equal(mask, fallback)


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


def test_sample_anchor_condition_validates_argument_types() -> None:
    with pytest.raises(TypeError, match="clean"):
        sample_anchor_condition([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="generator"):
        sample_anchor_condition(
            torch.zeros(1, 1, 4, 4),
            generator=object(),  # type: ignore[arg-type]
        )
