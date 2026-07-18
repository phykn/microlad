import unittest

import torch

from src.diffusion import DDPMProcess, DiffusionLoss, compute_loss


class AnchorRecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_fractions = None
        self.seen_axis_condition = None
        self.seen_anchor_image = None
        self.seen_anchor_mask = None

    def forward(
        self,
        x,
        t,
        fractions,
        axis_condition,
        *,
        anchor_image,
        anchor_mask,
    ):
        self.seen_fractions = fractions
        self.seen_axis_condition = axis_condition
        self.seen_anchor_image = anchor_image
        self.seen_anchor_mask = anchor_mask
        return torch.zeros_like(x)


class AnchorLossTest(unittest.TestCase):
    def test_loss_module_forwards_all_anchor_conditions(self):
        model = AnchorRecordingModel()
        clean = torch.zeros(2, 2, 8, 8)
        fractions = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
        axis_condition = torch.tensor([0, 2])
        anchor_image = torch.randn_like(clean)
        anchor_mask = torch.ones(2, 1, 8, 8)
        timesteps = torch.tensor([1, 2])
        ddpm = DDPMProcess(timesteps=4)

        DiffusionLoss(ddpm)(
            model,
            clean,
            fractions=fractions,
            t=timesteps,
            noise=torch.ones_like(clean),
            axis_condition=axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )

        self.assertIs(model.seen_fractions, fractions)
        self.assertIs(model.seen_axis_condition, axis_condition)
        self.assertIs(model.seen_anchor_image, anchor_image)
        self.assertIs(model.seen_anchor_mask, anchor_mask)

    def test_anchor_term_is_normalized_per_selected_sample(self):
        model = AnchorRecordingModel()
        clean = torch.zeros(2, 2, 8, 8)
        noise = torch.stack(
            [torch.ones(2, 8, 8), torch.full((2, 8, 8), 3.0)]
        )
        anchor_mask = torch.zeros(2, 1, 8, 8)
        anchor_mask[0, 0, 0, 0] = 1.0
        anchor_mask[1, 0, 4, 4] = 1.0

        loss, parts = compute_loss(
            model,
            DDPMProcess(timesteps=4),
            clean,
            t=torch.tensor([2, 3]),
            noise=noise,
            anchor_image=torch.zeros_like(clean),
            anchor_mask=anchor_mask,
            anchor_loss_weight=2.0,
        )

        self.assertTrue(torch.allclose(parts["noise"], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(parts["anchor"], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(loss, torch.tensor(15.0)))

    def test_empty_anchor_does_not_add_anchor_term(self):
        model = AnchorRecordingModel()
        clean = torch.zeros(2, 2, 8, 8)
        loss, parts = compute_loss(
            model,
            DDPMProcess(timesteps=4),
            clean,
            t=torch.tensor([0, 3]),
            noise=torch.ones_like(clean),
            anchor_image=torch.zeros_like(clean),
            anchor_mask=torch.zeros(2, 1, 8, 8),
            anchor_loss_weight=3.0,
        )

        self.assertNotIn("anchor", parts)
        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

if __name__ == "__main__":
    unittest.main()
