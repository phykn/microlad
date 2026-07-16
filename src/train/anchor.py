import math

import torch


def sample_anchor_condition(
    clean: torch.Tensor,
    *,
    empty_probability: float = 0.2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic image-space anchor conditions for MPDD training.

    The non-empty probability is split between line/segment, full/rectangle,
    and multiple/cross masks in a 4:3:1 ratio.  At the default empty
    probability this gives the intended 20/40/30/10 percent mixture.

    The clean image is returned unchanged.  Consumers must use the returned
    boolean mask to distinguish conditioned from unconditioned pixels.
    """
    if not isinstance(clean, torch.Tensor):
        raise TypeError("clean must be a torch.Tensor.")
    if clean.ndim != 4:
        raise ValueError("clean must have shape [batch, phases, height, width].")

    batch_size, num_phases, height, width = clean.shape
    if batch_size <= 0:
        raise ValueError("clean batch size must be positive.")
    if num_phases <= 0:
        raise ValueError("clean must contain at least one phase channel.")
    if height < 2 or width < 2:
        raise ValueError("clean height and width must both be at least 2.")

    if (
        not isinstance(empty_probability, (int, float))
        or isinstance(empty_probability, bool)
        or not math.isfinite(empty_probability)
        or not 0.0 <= empty_probability <= 1.0
    ):
        raise ValueError("empty_probability must be between zero and one.")
    empty_probability = float(empty_probability)

    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None.")
        generator_device = torch.device(generator.device)
        if generator_device.type != clean.device.type or (
            clean.device.type == "cuda"
            and generator_device.index is not None
            and clean.device.index is not None
            and generator_device.index != clean.device.index
        ):
            raise ValueError("generator and clean must use compatible devices.")

    device = clean.device
    random = torch.rand(
        batch_size,
        device=device,
        generator=generator,
    )
    remaining_probability = 1.0 - empty_probability
    line_end = empty_probability + 0.5 * remaining_probability
    face_end = empty_probability + 0.875 * remaining_probability
    categories = torch.zeros(batch_size, dtype=torch.int64, device=device)
    categories[random >= empty_probability] = 1
    categories[random >= line_end] = 2
    categories[random >= face_end] = 3

    y = torch.arange(height, device=device).view(1, height, 1)
    x = torch.arange(width, device=device).view(1, 1, width)

    rows = torch.randint(
        height,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    columns = torch.randint(
        width,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    horizontal = (y == rows).expand(-1, -1, width)
    vertical = (x == columns).expand(-1, height, -1)

    line_kinds = torch.randint(
        4,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    horizontal_lengths = (
        torch.randint(
            width - 1,
            (batch_size, 1, 1),
            device=device,
            generator=generator,
        )
        + 1
    )
    horizontal_starts = (
        (
            torch.rand(
                (batch_size, 1, 1),
                device=device,
                generator=generator,
            )
            * (width - horizontal_lengths + 1)
        )
        .floor()
        .to(torch.int64)
    )
    horizontal_segments = (
        (y == rows)
        & (x >= horizontal_starts)
        & (x < horizontal_starts + horizontal_lengths)
    )

    vertical_lengths = (
        torch.randint(
            height - 1,
            (batch_size, 1, 1),
            device=device,
            generator=generator,
        )
        + 1
    )
    vertical_starts = (
        (
            torch.rand(
                (batch_size, 1, 1),
                device=device,
                generator=generator,
            )
            * (height - vertical_lengths + 1)
        )
        .floor()
        .to(torch.int64)
    )
    vertical_segments = (
        (x == columns)
        & (y >= vertical_starts)
        & (y < vertical_starts + vertical_lengths)
    )

    line_masks = torch.where(
        line_kinds == 0,
        horizontal,
        torch.where(
            line_kinds == 1,
            vertical,
            torch.where(line_kinds == 2, horizontal_segments, vertical_segments),
        ),
    )

    rectangle_heights = (
        torch.randint(
            height,
            (batch_size, 1, 1),
            device=device,
            generator=generator,
        )
        + 1
    )
    rectangle_widths = (
        torch.randint(
            width,
            (batch_size, 1, 1),
            device=device,
            generator=generator,
        )
        + 1
    )
    accidentally_full = (rectangle_heights == height) & (rectangle_widths == width)
    rectangle_heights = torch.where(
        accidentally_full,
        rectangle_heights - 1,
        rectangle_heights,
    )
    rectangle_tops = (
        (
            torch.rand(
                (batch_size, 1, 1),
                device=device,
                generator=generator,
            )
            * (height - rectangle_heights + 1)
        )
        .floor()
        .to(torch.int64)
    )
    rectangle_lefts = (
        (
            torch.rand(
                (batch_size, 1, 1),
                device=device,
                generator=generator,
            )
            * (width - rectangle_widths + 1)
        )
        .floor()
        .to(torch.int64)
    )
    rectangles = (
        (y >= rectangle_tops)
        & (y < rectangle_tops + rectangle_heights)
        & (x >= rectangle_lefts)
        & (x < rectangle_lefts + rectangle_widths)
    )
    use_full_image = (
        torch.randint(
            2,
            (batch_size, 1, 1),
            device=device,
            generator=generator,
        )
        == 0
    )
    face_masks = torch.where(use_full_image, torch.ones_like(rectangles), rectangles)

    crosses = horizontal | vertical
    row_offsets = torch.randint(
        1,
        height,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    second_rows = (rows + row_offsets) % height
    parallel_horizontal = ((y == rows) | (y == second_rows)).expand(-1, -1, width)
    column_offsets = torch.randint(
        1,
        width,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    second_columns = (columns + column_offsets) % width
    parallel_vertical = ((x == columns) | (x == second_columns)).expand(-1, height, -1)
    multi_kinds = torch.randint(
        3,
        (batch_size, 1, 1),
        device=device,
        generator=generator,
    )
    multi_masks = torch.where(
        multi_kinds == 0,
        crosses,
        torch.where(multi_kinds == 1, parallel_horizontal, parallel_vertical),
    )

    masks = torch.zeros(
        (batch_size, height, width),
        dtype=torch.bool,
        device=device,
    )
    masks = torch.where((categories == 1)[:, None, None], line_masks, masks)
    masks = torch.where((categories == 2)[:, None, None], face_masks, masks)
    masks = torch.where((categories == 3)[:, None, None], multi_masks, masks)
    return clean, masks[:, None]
