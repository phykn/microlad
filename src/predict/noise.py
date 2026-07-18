import torch

from ..misc import require_finite
from .tiles import iter_tiles, make_window


def predict_tiles(
    model: torch.nn.Module,
    planes: torch.Tensor,
    steps: torch.Tensor,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
    fractions: torch.Tensor | None,
    axis_condition: int,
    guidance: float = 1.0,
    anchor_image: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _, _, height, width = planes.shape
    _validate_anchor_tiles(anchor_image, anchor_mask, planes)
    output = torch.zeros_like(planes)
    weights = torch.zeros(
        (1, 1, height, width),
        device=planes.device,
        dtype=planes.dtype,
    )
    window = (
        torch.ones(
            (1, 1, tile_size, tile_size),
            device=planes.device,
            dtype=planes.dtype,
        )
        if overlap == 0
        else make_window(
            tile_size,
            tile_size,
            device=planes.device,
            dtype=planes.dtype,
        ).view(1, 1, tile_size, tile_size)
    )

    for row, col in iter_tiles(
        height,
        width,
        tile_size=tile_size,
        overlap=overlap,
    ):
        for start in range(0, planes.shape[0], batch_size):
            stop = min(start + batch_size, planes.shape[0])
            patch = planes[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ]
            condition = _expand_fractions(fractions, patch.shape[0])
            axes = _expand_axis_condition(
                axis_condition,
                patch.shape[0],
                device=patch.device,
            )
            anchor_patch = (
                None
                if anchor_image is None
                else anchor_image[
                    start:stop,
                    :,
                    row : row + tile_size,
                    col : col + tile_size,
                ]
            )
            mask_patch = (
                None
                if anchor_mask is None
                else anchor_mask[
                    start:stop,
                    :,
                    row : row + tile_size,
                    col : col + tile_size,
                ]
            )
            noise = guide_noise(
                model,
                patch,
                steps[start:stop],
                condition=condition,
                axis_condition=axes,
                guidance=guidance,
                anchor_image=anchor_patch,
                anchor_mask=mask_patch,
            )
            if noise.shape != patch.shape:
                raise ValueError("model output must have the same shape as its input.")
            require_finite("predicted noise", noise)
            output[
                start:stop,
                :,
                row : row + tile_size,
                col : col + tile_size,
            ] += noise * window
        weights[:, :, row : row + tile_size, col : col + tile_size] += window

    return output / weights.clamp_min(torch.finfo(planes.dtype).tiny)


def guide_noise(
    model: torch.nn.Module,
    patch: torch.Tensor,
    steps: torch.Tensor,
    *,
    condition: torch.Tensor | None,
    axis_condition: torch.Tensor,
    guidance: float,
    anchor_image: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if (anchor_image is None) != (anchor_mask is None):
        raise ValueError("anchor_image and anchor_mask must be provided together.")
    return _guide_fraction_noise(
        model,
        patch,
        steps,
        condition=condition,
        axis_condition=axis_condition,
        guidance=guidance,
        anchor_image=anchor_image,
        anchor_mask=anchor_mask,
    )


def _guide_fraction_noise(
    model: torch.nn.Module,
    patch: torch.Tensor,
    steps: torch.Tensor,
    *,
    condition: torch.Tensor | None,
    axis_condition: torch.Tensor,
    guidance: float,
    anchor_image: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
) -> torch.Tensor:
    if condition is None:
        return _call_model(
            model,
            patch,
            steps,
            condition=None,
            axis_condition=axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
    if guidance == 1.0:
        return _call_model(
            model,
            patch,
            steps,
            condition=condition,
            axis_condition=axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )

    model_input = torch.cat([patch, patch], dim=0)
    model_steps = torch.cat([steps, steps], dim=0)
    model_cond = torch.cat([torch.zeros_like(condition), condition], dim=0)
    model_anchor = (
        None
        if anchor_image is None
        else torch.cat([anchor_image, anchor_image], dim=0)
    )
    model_mask = (
        None if anchor_mask is None else torch.cat([anchor_mask, anchor_mask], dim=0)
    )
    model_axis = torch.cat([axis_condition, axis_condition], dim=0)
    prediction = _call_model(
        model,
        model_input,
        model_steps,
        condition=model_cond,
        axis_condition=model_axis,
        anchor_image=model_anchor,
        anchor_mask=model_mask,
    )
    null, guided = prediction.chunk(2, dim=0)
    return null + guidance * (guided - null)


def _call_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    steps: torch.Tensor,
    *,
    condition: torch.Tensor | None,
    axis_condition: torch.Tensor,
    anchor_image: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
) -> torch.Tensor:
    return model(
        image,
        steps,
        condition,
        axis_condition,
        anchor_image=anchor_image,
        anchor_mask=anchor_mask,
    )


def _expand_fractions(
    fractions: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    return None if fractions is None else fractions[None].expand(batch_size, -1)


def _expand_axis_condition(
    axis_condition: int,
    batch_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.full(
        (batch_size,),
        axis_condition,
        dtype=torch.long,
        device=device,
    )


def _validate_anchor_tiles(
    anchor_image: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
    planes: torch.Tensor,
) -> None:
    if (anchor_image is None) != (anchor_mask is None):
        raise ValueError("anchor_image and anchor_mask must be provided together.")
    if anchor_image is None or anchor_mask is None:
        return
    if anchor_image.shape != planes.shape:
        raise ValueError("anchor_image must have the same shape as planes.")
    if anchor_mask.shape != (planes.shape[0], 1, *planes.shape[-2:]):
        raise ValueError("anchor_mask must have shape [B, 1, H, W].")
