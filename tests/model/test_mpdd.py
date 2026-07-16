import unittest

import torch

from src.model import MPDDUNet, encode_labels


class RepresentationTest(unittest.TestCase):
    def test_encode_labels_preserves_categorical_values(self):
        labels = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.float32)

        encoded = encode_labels(labels, num_phases=3)

        self.assertEqual(encoded.shape, torch.Size([1, 3, 2, 2]))
        self.assertTrue(encoded.is_contiguous())
        self.assertTrue(torch.equal(encoded.argmax(dim=1), labels[:, 0].long()))
        self.assertEqual(set(encoded.unique().tolist()), {-1.0, 1.0})


class MPDDUNetTest(unittest.TestCase):
    def test_forward_uses_low_resolution_attention_and_preserves_shape(self):
        model = MPDDUNet(
            num_phases=3,
            image_size=8,
            base_ch=4,
            time_dim=8,
        )
        attention = [
            module
            for module in model.modules()
            if type(module).__name__ == "SelfAttention"
        ]

        output = model(
            torch.randn(2, 3, 8, 8),
            torch.tensor([0, 1]),
            torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]),
        )

        self.assertEqual(len(attention), 3)
        self.assertEqual(output.shape, torch.Size([2, 3, 8, 8]))

    def test_rejects_non_normalized_fraction_condition(self):
        model = MPDDUNet(num_phases=2, image_size=8, base_ch=4, time_dim=8)

        with self.assertRaisesRegex(ValueError, "sum to one"):
            model(
                torch.randn(1, 2, 8, 8),
                torch.tensor([0]),
                torch.tensor([[0.2, 0.2]]),
            )

    def test_axis_conditioning_gives_all_embedding_rows_gradients(self):
        model = MPDDUNet(
            num_phases=2,
            image_size=8,
            base_ch=4,
            time_dim=8,
            num_axis_conditions=3,
        )

        image = torch.randn(1, 2, 8, 8).expand(3, -1, -1, -1).clone()
        output = model(
            image,
            torch.tensor([1, 1, 1]),
            torch.tensor([[0.5, 0.5]]).expand(3, -1),
            torch.tensor([0, 1, 2]),
        )
        output.square().mean().backward()

        self.assertEqual(model.axis_emb.weight.shape, torch.Size([3, 8]))
        self.assertFalse(torch.allclose(output[0], output[1]))
        self.assertFalse(torch.allclose(output[1], output[2]))
        self.assertIsNotNone(model.axis_emb.weight.grad)
        self.assertTrue(torch.all(model.axis_emb.weight.grad.abs().sum(dim=1) > 0))

    def test_axis_conditioning_validates_shape_dtype_and_range(self):
        model = MPDDUNet(
            num_phases=2,
            image_size=8,
            base_ch=4,
            time_dim=8,
            num_axis_conditions=3,
        )
        image = torch.randn(2, 2, 8, 8)
        timestep = torch.tensor([0, 1])

        with self.assertRaisesRegex(ValueError, "required"):
            model(image, timestep)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(image, timestep, axis_condition=torch.tensor([[0], [1]]))
        with self.assertRaisesRegex(TypeError, "torch.long"):
            model(image, timestep, axis_condition=torch.tensor([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "range"):
            model(image, timestep, axis_condition=torch.tensor([0, 3]))

    def test_legacy_model_has_no_axis_state_and_strict_loads(self):
        model = MPDDUNet(num_phases=2, image_size=8, base_ch=4, time_dim=8)
        state = model.state_dict()
        restored = MPDDUNet(num_phases=2, image_size=8, base_ch=4, time_dim=8)

        restored.load_state_dict(state, strict=True)

        self.assertFalse(any(name.startswith("axis_emb.") for name in state))
        with self.assertRaisesRegex(ValueError, "num_axis_conditions"):
            restored(
                torch.randn(1, 2, 8, 8),
                torch.tensor([0]),
                axis_condition=torch.tensor([0]),
            )

    def test_num_axis_conditions_only_accepts_legacy_or_three_axes(self):
        with self.assertRaisesRegex(ValueError, "either 0 or 3"):
            MPDDUNet(
                num_phases=2,
                image_size=8,
                base_ch=4,
                time_dim=8,
                num_axis_conditions=2,
            )


if __name__ == "__main__":
    unittest.main()
