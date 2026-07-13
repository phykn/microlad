import unittest
from unittest.mock import patch

import torch

from src.app.api import CriticConfig
from src.modeling.critic import LatentCritic
from src.pipelines.guidance.critic.data import encode_refs
from src.pipelines.guidance.critic.train import train_critic


class IdentityVAE(torch.nn.Module):
    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.repeat(1, 2, 1, 1), torch.zeros_like(image).repeat(1, 2, 1, 1)


class FakeProgress:
    instances = []

    def __init__(self, iterable, **kwargs) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.postfixes = []
        self.__class__.instances.append(self)

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, values) -> None:
        self.postfixes.append(values)


class CriticGuidanceTest(unittest.TestCase):
    def test_reference_augmentation_happens_before_vae_encoding(self):
        labels = torch.arange(16 * 16).reshape(1, 16, 16).float()

        bank = encode_refs(IdentityVAE(), labels, batch_size=3)

        self.assertEqual(bank.shape, torch.Size([8, 2, 16, 16]))
        self.assertTrue(torch.equal(bank[0, 0].flatten().sort().values, labels.flatten()))

    def test_critic_warmup_freezes_parameters_and_validates_input_gradient(self):
        torch.manual_seed(0)
        critic = LatentCritic(2, base_ch=4)
        real = torch.randn(12, 2, 16, 16)
        fake = torch.randn(2, 2, 16, 16, 16)
        config = CriticConfig(steps=1, batch_size=2, candidate_count=2)

        stats = train_critic(critic, real, fake, config=config)

        self.assertEqual(int(stats["critic_steps"]), 1)
        self.assertTrue(bool(stats["critic_input_gradient_finite"]))
        self.assertTrue(torch.isfinite(stats["critic_input_gradient_norm"]))
        self.assertTrue(all(not parameter.requires_grad for parameter in critic.parameters()))
        self.assertFalse(critic.training)

    def test_critic_progress_shows_loss_and_margin(self):
        torch.manual_seed(0)
        critic = LatentCritic(2, base_ch=4)
        real = torch.randn(12, 2, 16, 16)
        fake = torch.randn(2, 2, 16, 16, 16)
        config = CriticConfig(steps=1, batch_size=2, candidate_count=2)
        FakeProgress.instances = []

        with patch(
            "src.pipelines.guidance.critic.train.tqdm",
            FakeProgress,
        ):
            train_critic(critic, real, fake, config=config, progress=True)

        progress = FakeProgress.instances[0]
        self.assertEqual(progress.kwargs["desc"], "Latent critic")
        self.assertEqual(set(progress.postfixes[-1]), {"loss", "margin"})


if __name__ == "__main__":
    unittest.main()
