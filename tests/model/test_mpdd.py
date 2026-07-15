import unittest

import torch

from src.model import MPDDUNet, encode_labels


class RepresentationTest(unittest.TestCase):
    def test_encode_labels_preserves_categorical_values(self):
        labels = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.float32)

        encoded = encode_labels(labels, num_phases=3)

        self.assertEqual(encoded.shape, torch.Size([1, 3, 2, 2]))
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


if __name__ == "__main__":
    unittest.main()
