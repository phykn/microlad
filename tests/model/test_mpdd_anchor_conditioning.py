import unittest

import torch

from src.model import MPDDUNet


def _make_model(**kwargs) -> MPDDUNet:
    torch.manual_seed(13)
    return MPDDUNet(
        num_phases=2,
        image_size=8,
        base_ch=4,
        time_dim=8,
        **kwargs,
    ).eval()


class AnchorConditioningTest(unittest.TestCase):
    def test_legacy_model_has_no_extra_state_and_strict_loads(self):
        legacy = _make_model()
        explicit_legacy = _make_model(anchor_conditioning=False)
        state = legacy.state_dict()

        explicit_legacy.load_state_dict(state, strict=True)

        self.assertEqual(list(state), list(explicit_legacy.state_dict()))
        self.assertEqual(
            sum(parameter.numel() for parameter in legacy.parameters()),
            sum(parameter.numel() for parameter in explicit_legacy.parameters()),
        )
        self.assertFalse(any("anchor" in name for name in state))
        self.assertEqual(legacy.enc1.block1.conv1.in_channels, 2)

    def test_empty_or_released_anchor_is_exactly_the_null_condition(self):
        model = _make_model(anchor_conditioning=True, anchor_release_step=5)
        image = torch.randn(2, 2, 8, 8)
        anchor_image = torch.randn_like(image)
        empty_mask = torch.zeros(2, 1, 8, 8)
        full_mask = torch.ones_like(empty_mask)

        with torch.no_grad():
            active_null = model(image, torch.tensor([5, 5]))
            active_empty = model(
                image,
                torch.tensor([5, 5]),
                anchor_image=anchor_image,
                anchor_mask=empty_mask,
            )
            active_anchor = model(
                image,
                torch.tensor([5, 5]),
                anchor_image=anchor_image,
                anchor_mask=full_mask,
            )
            released_null = model(image, torch.tensor([4, 4]))
            released_anchor = model(
                image,
                torch.tensor([4, 4]),
                anchor_image=anchor_image,
                anchor_mask=full_mask,
            )

        self.assertTrue(torch.equal(active_null, active_empty))
        self.assertTrue(torch.equal(active_null, active_anchor))
        self.assertTrue(torch.equal(released_null, released_anchor))

    def test_active_anchor_changes_output_after_output_projection_is_perturbed(self):
        model = _make_model(anchor_conditioning=True, anchor_release_step=5)
        image = torch.randn(2, 2, 8, 8)
        anchor_image = torch.ones_like(image)
        anchor_mask = torch.ones(2, 1, 8, 8)
        projection = model.anchor_encoder.outputs[0]

        with torch.no_grad():
            projection.weight.fill_(0.5)
            projection.bias.fill_(0.25)
            null_output = model(image, torch.tensor([5, 5]))
            anchor_output = model(
                image,
                torch.tensor([5, 5]),
                anchor_image=anchor_image,
                anchor_mask=anchor_mask,
            )

        self.assertFalse(torch.allclose(null_output, anchor_output))

    def test_anchor_inputs_must_be_paired_and_match_full_resolution(self):
        model = _make_model(anchor_conditioning=True)
        image = torch.randn(1, 2, 8, 8)
        timestep = torch.tensor([0])
        anchor_image = torch.randn_like(image)
        anchor_mask = torch.ones(1, 1, 8, 8)

        with self.assertRaisesRegex(ValueError, "provided together"):
            model(image, timestep, anchor_image=anchor_image)
        with self.assertRaisesRegex(ValueError, "provided together"):
            model(image, timestep, anchor_mask=anchor_mask)
        with self.assertRaisesRegex(ValueError, "same shape"):
            model(
                image,
                timestep,
                anchor_image=anchor_image[:, :, :-1],
                anchor_mask=anchor_mask,
            )
        with self.assertRaisesRegex(ValueError, r"\[B, 1, H, W\]"):
            model(
                image,
                timestep,
                anchor_image=anchor_image,
                anchor_mask=anchor_mask[:, :, :-1],
            )

    def test_axis_fraction_and_anchor_conditions_work_together(self):
        model = _make_model(
            num_axis_conditions=3,
            anchor_conditioning=True,
            anchor_release_step=3,
        )
        image = torch.randn(3, 2, 8, 8)
        fractions = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
        axis_condition = torch.tensor([0, 1, 2])

        output = model(
            image,
            torch.tensor([3, 4, 5]),
            fractions,
            axis_condition,
            anchor_image=torch.randn_like(image),
            anchor_mask=torch.ones(3, 1, 8, 8),
        )

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
