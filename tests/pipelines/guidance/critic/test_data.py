import unittest

import torch

from src.pipelines.guidance.critic.data import (
    encode_refs,
    split_fake_bank,
    split_real_bank,
)


class IdentityVAE(torch.nn.Module):
    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repeated = image.repeat(1, 2, 1, 1)
        return repeated, torch.zeros_like(repeated)


class CriticDataTest(unittest.TestCase):
    def test_reference_augmentation_happens_before_vae_encoding(self):
        labels = torch.arange(16 * 16).reshape(1, 16, 16).float()

        bank = encode_refs(IdentityVAE(), labels, batch_size=3)

        self.assertEqual(bank.shape, torch.Size([1, 8, 2, 16, 16]))
        self.assertTrue(
            torch.equal(bank[0, 0, 0].flatten().sort().values, labels.flatten())
        )

    def test_real_sources_do_not_cross_train_validation(self):
        sources = torch.arange(4).view(4, 1, 1, 1, 1).expand(4, 8, 2, 16, 16)

        train, validation, source_holdout = split_real_bank(
            sources.float(),
            validation_size=2,
        )

        self.assertTrue(source_holdout)
        self.assertTrue(
            set(train[:, 0, 0, 0].tolist()).isdisjoint(
                validation[:, 0, 0, 0].tolist()
            )
        )

    def test_fake_volumes_do_not_cross_train_validation(self):
        volumes = torch.ones(3, 2, 16, 16, 16)
        volumes[0] = 0.0

        train, validation = split_fake_bank(volumes, crop_size=16)

        self.assertTrue(torch.all(train == 1.0))
        self.assertTrue(torch.all(validation == 0.0))


if __name__ == "__main__":
    unittest.main()
