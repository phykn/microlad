import tempfile

import torch

from src.diffusion import DDPMProcess, DiffusionLoss
from src.model import MPDDUNet
from src.train import MPDDTrainer


def test_mpdd_and_internal_anchor_encoder_train_and_checkpoint_together() -> None:
    torch.manual_seed(5)
    model = MPDDUNet(
        num_phases=2,
        image_size=8,
        base_ch=4,
        time_dim=8,
    )
    images = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float32)
    fractions = torch.tensor([[0.5, 0.5]]).expand(3, -1)
    axes = torch.tensor([0, 1, 2])
    base_before = model.out.weight.detach().clone()
    anchor_before = model.anchor_encoder.input.weight.detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        trainer = MPDDTrainer(
            model=model,
            loader=[(images, fractions, axes)],
            loss=DiffusionLoss(
                DDPMProcess(timesteps=2, beta_start=0.01, beta_end=0.02),
                anchor_loss_weight=0.25,
            ),
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            num_phases=2,
            steps=2,
            device="cpu",
            run_root=tmp,
            anchor_empty_probability=0.0,
            ema_decay=0.0,
        )

        first = trainer.train_step()
        second = trainer.train_step()
        checkpoint = torch.load(
            trainer.last_weight_dir / "model.pt",
            weights_only=True,
        )
        trainer.close()

    assert first["anchor_coverage"] > 0.0
    assert "anchor" in first and "anchor" in second
    assert not torch.equal(base_before, model.out.weight)
    assert not torch.equal(anchor_before, model.anchor_encoder.input.weight)
    assert any(name.startswith("anchor_encoder.") for name in checkpoint["model"])
    assert checkpoint["step"] == 2
