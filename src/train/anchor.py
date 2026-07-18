import math

import torch


def sample_anchor_condition(
    clean: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample equally likely empty, crop, segment, and multiple anchors."""
    if not isinstance(clean, torch.Tensor):
        raise TypeError("clean must be a torch.Tensor.")
    if clean.ndim != 4:
        raise ValueError("clean must have shape [batch, phases, height, width].")

    b, c, h, w = clean.shape
    if b <= 0:
        raise ValueError("clean batch size must be positive.")
    if c <= 0:
        raise ValueError("clean must contain at least one phase channel.")
    if h < 2 or w < 2:
        raise ValueError("clean height and width must both be at least 2.")
    _check_generator(generator, clean.device)

    dev = clean.device
    kind = torch.randint(4, (b,), device=dev, generator=generator)
    crop1 = _crops(b, h, w, device=dev, generator=generator)
    crop2 = _crops(b, h, w, device=dev, generator=generator)
    seg1 = _segments(b, h, w, device=dev, generator=generator)
    seg2 = _segments(b, h, w, device=dev, generator=generator)
    pick1 = torch.rand(b, 1, 1, device=dev, generator=generator) < 0.5
    pick2 = torch.rand(b, 1, 1, device=dev, generator=generator) < 0.5
    a = torch.where(pick1, crop1, seg1)
    b2 = torch.where(pick2, crop2, seg2)
    cross = _crosses(b, h, w, device=dev, generator=generator)
    multi = _merge_multiple(a, b2, cross)

    mask = torch.zeros((b, h, w), dtype=torch.bool, device=dev)
    mask = torch.where((kind == 1)[:, None, None], crop1, mask)
    mask = torch.where((kind == 2)[:, None, None], seg1, mask)
    mask = torch.where((kind == 3)[:, None, None], multi, mask)
    return clean, mask[:, None]


def _merge_multiple(
    a: torch.Tensor,
    b: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    a_extra = (a & ~b).flatten(start_dim=1).any(dim=1)
    b_extra = (b & ~a).flatten(start_dim=1).any(dim=1)
    valid = a_extra & b_extra
    return torch.where(valid[:, None, None], a | b, fallback)


def _crosses(
    b: int,
    h: int,
    w: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    y = torch.arange(h, device=device).view(1, h, 1)
    x = torch.arange(w, device=device).view(1, 1, w)
    row = torch.randint(h, (b, 1, 1), device=device, generator=generator)
    col = torch.randint(w, (b, 1, 1), device=device, generator=generator)
    return (y == row) | (x == col)


def _crops(
    b: int,
    h: int,
    w: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    m = min(h, w)
    lo = max(math.ceil(m * 0.25), 1)
    size = torch.randint(lo, m + 1, (b, 1, 1), device=device, generator=generator)
    top = (
        torch.rand(b, 1, 1, device=device, generator=generator)
        * (h - size + 1)
    ).floor().to(torch.long)
    left = (
        torch.rand(b, 1, 1, device=device, generator=generator)
        * (w - size + 1)
    ).floor().to(torch.long)
    y = torch.arange(h, device=device).view(1, h, 1)
    x = torch.arange(w, device=device).view(1, 1, w)
    return (y >= top) & (y < top + size) & (x >= left) & (x < left + size)


def _segments(
    b: int,
    h: int,
    w: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    y = torch.arange(h, device=device).view(1, h, 1)
    x = torch.arange(w, device=device).view(1, 1, w)
    row = torch.randint(h, (b, 1, 1), device=device, generator=generator)
    col = torch.randint(w, (b, 1, 1), device=device, generator=generator)

    h_lo = max(math.ceil(w * 0.25), 1)
    h_len = torch.randint(h_lo, w + 1, (b, 1, 1), device=device, generator=generator)
    h_start = (
        torch.rand(b, 1, 1, device=device, generator=generator)
        * (w - h_len + 1)
    ).floor().to(torch.long)
    h_seg = (y == row) & (x >= h_start) & (x < h_start + h_len)

    v_lo = max(math.ceil(h * 0.25), 1)
    v_len = torch.randint(v_lo, h + 1, (b, 1, 1), device=device, generator=generator)
    v_start = (
        torch.rand(b, 1, 1, device=device, generator=generator)
        * (h - v_len + 1)
    ).floor().to(torch.long)
    v_seg = (x == col) & (y >= v_start) & (y < v_start + v_len)
    horizontal = torch.rand(b, 1, 1, device=device, generator=generator) < 0.5
    return torch.where(horizontal, h_seg, v_seg)


def _check_generator(
    generator: torch.Generator | None,
    device: torch.device,
) -> None:
    if generator is None:
        return
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")
    gen = torch.device(generator.device)
    if gen.type != device.type or (
        device.type == "cuda"
        and gen.index is not None
        and device.index is not None
        and gen.index != device.index
    ):
        raise ValueError("generator and clean must use compatible devices.")
