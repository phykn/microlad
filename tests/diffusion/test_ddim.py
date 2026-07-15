import unittest

import torch

from src.diffusion import DDIMProcess, DDPMProcess


class DDIMProcessTest(unittest.TestCase):
    def test_schedule_spans_full_training_schedule(self):
        ddpm = DDPMProcess(timesteps=10, beta_start=0.01, beta_end=0.02)

        self.assertEqual(
            DDIMProcess(ddpm, sampling_steps=4).schedule,
            [(9, 6), (6, 3), (3, 0), (0, -1)],
        )
        self.assertEqual(
            DDIMProcess(ddpm, sampling_steps=1).schedule,
            [(9, -1)],
        )

    def test_step_and_renoise_use_cumulative_alpha(self):
        ddpm = DDPMProcess(timesteps=10, beta_start=0.01, beta_end=0.02)
        ddim = DDIMProcess(ddpm, sampling_steps=4)
        clean = torch.full((2, 3, 4, 4), 0.25)
        noise = torch.full_like(clean, 0.5)
        timesteps = torch.full((2,), 9, dtype=torch.long)
        noisy = ddpm.add_noise(clean, timesteps, noise=noise)

        previous = ddim.step(
            noisy,
            noise,
            step=9,
            prev_step=3,
        )
        restored = ddim.renoise(
            previous,
            source_step=3,
            target_step=9,
            noise=noise,
        )

        alpha = ddpm.alphas_cumprod[3]
        expected_previous = torch.sqrt(alpha) * clean + torch.sqrt(1 - alpha) * noise
        ratio = ddpm.alphas_cumprod[9] / ddpm.alphas_cumprod[3]
        expected_restored = (
            torch.sqrt(ratio) * expected_previous + torch.sqrt(1 - ratio) * noise
        )
        self.assertTrue(torch.allclose(previous, expected_previous))
        self.assertTrue(torch.allclose(restored, expected_restored))

    def test_validates_sampling_steps(self):
        ddpm = DDPMProcess(timesteps=10, beta_start=0.01, beta_end=0.02)

        with self.assertRaisesRegex(ValueError, "sampling_steps"):
            DDIMProcess(ddpm, sampling_steps=0)


if __name__ == "__main__":
    unittest.main()
