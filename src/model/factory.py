from collections.abc import Mapping
from typing import Any

from .mpdd import MPDDUNet


_REQUIRED_CONFIG_KEYS = (
    "size",
    "num_phases",
    "base_ch",
    "time_dim",
)


def build_mpdd_model(
    values: Mapping[str, Any],
    *,
    label: str = "model config",
) -> MPDDUNet:
    """Build one MPDD architecture from a saved or training config."""

    missing = [name for name in _REQUIRED_CONFIG_KEYS if name not in values]
    if missing:
        raise ValueError(f"{label} is missing required value: {', '.join(missing)}")
    return MPDDUNet(
        num_phases=values["num_phases"],
        image_size=values["size"],
        base_ch=values["base_ch"],
        time_dim=values["time_dim"],
        num_axis_conditions=values.get("num_axis_conditions", 0),
        anchor_conditioning=values.get("anchor_conditioning", False),
        anchor_release_step=values.get("anchor_release_step", 0),
    )
