from pathlib import Path

import torch
import yaml

from src.model import MPDDUNet
from src.predict import load_predictor


def test_anchor_conditioned_checkpoint_round_trips_as_one_model(
    tmp_path: Path,
) -> None:
    config = {
        "data": {"size": 8, "num_phases": 2},
        "model": {
            "base_ch": 4,
            "time_dim": 8,
        },
        "diffusion": {
            "timesteps": 4,
            "beta_start": 0.01,
            "beta_end": 0.02,
        },
    }
    (tmp_path / "model.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "weight" / "mpdd" / "last" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    model = MPDDUNet(
        num_phases=2,
        image_size=8,
        base_ch=4,
        time_dim=8,
    )
    torch.save({"model": model.state_dict()}, checkpoint)

    predictor = load_predictor(tmp_path, device="cpu")
    restored = predictor.sampler.model

    assert any(name.startswith("anchor_encoder.") for name in restored.state_dict())
