import unittest

import torch

from src.diffusion import DDPMProcess, DiffusionLoss, compute_loss


class AnchorRecordingModel(torch.nn.Module):
    def __init__(self, anchor_release_step: int = 0) -> None:
        super().__init__()
        self.anchor_release_step = anchor_release_step
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


class PrescribedNoiseModel(AnchorRecordingModel):
    def __init__(self, prediction, anchor_release_step: int = 0) -> None:
        super().__init__(anchor_release_step=anchor_release_step)
        self.prediction = prediction

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
        super().forward(
            x,
            t,
            fractions,
            axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        return self.prediction.to(x)


class TrainableNoiseModel(AnchorRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

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
        return torch.zeros_like(x) + self.bias


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
        model = AnchorRecordingModel(anchor_release_step=2)
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

    def test_empty_or_released_anchor_does_not_add_anchor_term(self):
        model = AnchorRecordingModel(anchor_release_step=2)
        clean = torch.zeros(2, 2, 8, 8)
        cases = (
            ("empty", torch.tensor([2, 3]), torch.zeros(2, 1, 8, 8)),
            ("released", torch.tensor([0, 1]), torch.ones(2, 1, 8, 8)),
        )

        for name, timesteps, anchor_mask in cases:
            with self.subTest(name=name):
                loss, parts = compute_loss(
                    model,
                    DDPMProcess(timesteps=4),
                    clean,
                    t=timesteps,
                    noise=torch.ones_like(clean),
                    anchor_image=torch.zeros_like(clean),
                    anchor_mask=anchor_mask,
                    anchor_loss_weight=3.0,
                )

                self.assertNotIn("anchor", parts)
                self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_correct_phase_prediction_beats_phase_flip(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.zeros(1, 2, 4, 4)
        clean[:, 0] = 1.0
        flipped = clean.flip(dims=(1,))
        noise = torch.zeros_like(clean)
        t = torch.tensor([2])
        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
        alpha = ddpm.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
        flip_prediction = alpha * (clean - flipped) / sigma
        anchor_mask = torch.ones(1, 1, 4, 4)

        _, correct_parts = compute_loss(
            PrescribedNoiseModel(noise),
            ddpm,
            clean,
            t=t,
            noise=noise,
            anchor_image=clean,
            anchor_mask=anchor_mask,
            anchor_phase_loss_weight=1.0,
        )
        _, flipped_parts = compute_loss(
            PrescribedNoiseModel(flip_prediction),
            ddpm,
            clean,
            t=t,
            noise=noise,
            anchor_image=clean,
            anchor_mask=anchor_mask,
            anchor_phase_loss_weight=1.0,
        )

        self.assertLess(
            correct_parts["anchor_phase"],
            flipped_parts["anchor_phase"],
        )

    def test_phase_term_is_normalized_per_selected_sample(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.zeros(2, 2, 4, 4)
        noise = torch.zeros_like(clean)
        t = torch.tensor([2, 2])
        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(2, 1, 1, 1)
        logits = torch.zeros_like(clean)
        logits[1, 0] = 2.0
        prediction = -logits / sigma
        anchor_image = torch.zeros_like(clean)
        anchor_image[:, 1] = 1.0
        anchor_mask = torch.zeros(2, 1, 4, 4)
        anchor_mask[0, 0, 0, 0] = 1.0
        anchor_mask[1, 0] = 1.0

        _, parts = compute_loss(
            PrescribedNoiseModel(prediction),
            ddpm,
            clean,
            t=t,
            noise=noise,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
            anchor_phase_loss_weight=1.0,
        )

        expected = torch.stack(
            [
                torch.log(torch.tensor(2.0)),
                torch.log1p(torch.exp(torch.tensor(2.0))),
            ]
        ).mean()
        self.assertTrue(torch.allclose(parts["anchor_phase"], expected))

    def test_empty_or_released_anchor_does_not_add_phase_term(self):
        model = AnchorRecordingModel(anchor_release_step=2)
        clean = torch.zeros(2, 2, 8, 8)
        clean[:, 0] = 1.0
        cases = (
            ("empty", torch.tensor([2, 3]), torch.zeros(2, 1, 8, 8)),
            ("released", torch.tensor([0, 1]), torch.ones(2, 1, 8, 8)),
        )

        for name, timesteps, anchor_mask in cases:
            with self.subTest(name=name):
                loss, parts = compute_loss(
                    model,
                    DDPMProcess(timesteps=4),
                    clean,
                    t=timesteps,
                    noise=torch.ones_like(clean),
                    anchor_image=clean,
                    anchor_mask=anchor_mask,
                    anchor_phase_loss_weight=3.0,
                )

                self.assertNotIn("anchor_phase", parts)
                self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_phase_loss_gradient_is_finite_at_high_noise(self):
        model = TrainableNoiseModel()
        ddpm = DDPMProcess(timesteps=1000)
        clean = torch.zeros(1, 2, 4, 4)
        clean[:, 0] = 1.0

        loss, parts = compute_loss(
            model,
            ddpm,
            clean,
            t=torch.tensor([999]),
            noise=torch.ones_like(clean),
            anchor_image=clean,
            anchor_mask=torch.ones(1, 1, 4, 4),
            anchor_phase_loss_weight=1.0,
        )
        loss.backward()

        self.assertIn("anchor_phase", parts)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.isfinite(model.bias.grad))

    def test_phase_loss_weight_must_be_finite_and_non_negative(self):
        ddpm = DDPMProcess(timesteps=4)
        clean = torch.zeros(1, 2, 4, 4)
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    DiffusionLoss(ddpm, anchor_phase_loss_weight=value)
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    compute_loss(
                        AnchorRecordingModel(),
                        ddpm,
                        clean,
                        t=torch.tensor([0]),
                        noise=torch.zeros_like(clean),
                        anchor_phase_loss_weight=value,
                    )


if __name__ == "__main__":
    unittest.main()
