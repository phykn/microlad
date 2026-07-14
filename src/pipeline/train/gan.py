from collections.abc import Iterable
from pathlib import Path

import torch
from tqdm import tqdm

from src.modeling.latent_gan import critic_loss, gradient_penalty, guidance_loss
from src.pipeline.predict.guidance.latent_slices import sample_slices
from src.pipeline.train.misc.distributed import is_main_process, unwrap_model
from src.pipeline.train.misc.run import log_stats, setup_run_dirs, write_checkpoint


class GANTrainer:
    def __init__(
        self,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        vae: torch.nn.Module,
        dataloader: Iterable,
        fake_dataloader: Iterable,
        generator_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        *,
        steps: int,
        critic_steps: int,
        gp_weight: float,
        clip_grad_norm: float | None,
        save_every: int,
        device: str | torch.device,
        run_root: str | Path = "run",
    ) -> None:
        if steps <= 0 or critic_steps <= 0 or save_every <= 0:
            raise ValueError("training and save counts must be positive.")
        if gp_weight < 0.0:
            raise ValueError("gp_weight must be non-negative.")
        if clip_grad_norm is not None and clip_grad_norm <= 0.0:
            raise ValueError("clip_grad_norm must be positive or None.")

        self.device = torch.device(device)
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.vae = vae.to(self.device)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.fake_dataloader = fake_dataloader
        self.fake_iterator = iter(fake_dataloader)
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.steps = steps
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight
        self.clip_grad_norm = clip_grad_norm
        self.save_every = save_every
        self.step = 0
        self.axis_offset = 0
        self.is_main_process = is_main_process()
        (
            self.run_dir,
            self.log_dir,
            self.weight_dir,
            self.last_weight_dir,
            self.writer,
        ) = setup_run_dirs(
            run_root=run_root,
            component="gan",
            is_main_process=self.is_main_process,
        )
        self._save()

    def train_step(self) -> dict[str, float]:
        self.generator.train()
        self.critic.train()
        critic_losses = []
        penalties = []
        margins = []
        batch_size = None
        latent_dtype = None

        for _ in range(self.critic_steps):
            images = self._next_images()
            with torch.no_grad():
                real, _ = self.vae.encode(images)
                batch_size = int(real.shape[0])
                latent_dtype = real.dtype
                noise = torch.randn(
                    batch_size,
                    int(unwrap_model(self.generator).noise_ch),
                    device=self.device,
                    dtype=latent_dtype,
                )
                generator_fake = self.generator(noise)
                volumes = self._next_fake_volumes().to(
                    device=self.device,
                    dtype=real.dtype,
                )
                lmpdd_fake = sample_slices(
                    volumes,
                    count=batch_size,
                    crop_size=int(real.shape[-1]),
                    axis_offset=self.axis_offset,
                )
                self.axis_offset = (self.axis_offset + int(real.shape[0])) % 3

            self.critic_optimizer.zero_grad(set_to_none=True)
            real_scores = self.critic(real)
            fake_scores = torch.cat(
                (self.critic(generator_fake), self.critic(lmpdd_fake)),
                dim=0,
            )
            penalty = 0.5 * (
                gradient_penalty(self.critic, real, generator_fake)
                + gradient_penalty(self.critic, real, lmpdd_fake)
            )
            loss = critic_loss(
                real_scores,
                fake_scores,
                penalty,
                gp_weight=self.gp_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("GAN critic produced a non-finite loss.")
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.clip_grad_norm,
                )
            self.critic_optimizer.step()
            critic_losses.append(loss.detach())
            penalties.append(penalty.detach())
            margins.append((real_scores.mean() - fake_scores.mean()).detach())

        assert batch_size is not None and latent_dtype is not None
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(
            batch_size,
            int(unwrap_model(self.generator).noise_ch),
            device=self.device,
            dtype=latent_dtype,
        )
        generator_fake = self.generator(noise)
        adversarial = guidance_loss(self.critic(generator_fake))
        if not torch.isfinite(adversarial):
            raise RuntimeError("GAN generator produced a non-finite loss.")
        adversarial.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.clip_grad_norm,
            )
        self.generator_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        self.step += 1
        stats = {
            "critic_loss": float(torch.stack(critic_losses).mean().cpu()),
            "critic_margin": float(torch.stack(margins).mean().cpu()),
            "gradient_penalty": float(torch.stack(penalties).mean().cpu()),
            "generator_loss": float(adversarial.detach().cpu()),
        }
        log_stats(self.writer, stats, self.step)
        self._save()
        return stats

    def train(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        progress = tqdm(
            range(self.steps),
            total=self.steps,
            desc="gan",
            disable=not self.is_main_process,
        )
        for _ in progress:
            stats = self.train_step()
            progress.set_postfix(
                {name: f"{value:.4g}" for name, value in stats.items()}
            )
        return stats

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def _next_images(self) -> torch.Tensor:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            try:
                batch = next(self.iterator)
            except StopIteration as exc:
                raise ValueError("dataloader cannot provide GAN training batches.") from exc
        if isinstance(batch, torch.Tensor):
            images = batch
        elif isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            images = batch[0]
        else:
            raise TypeError(
                "batch must be a tensor or a tuple/list whose first item is a tensor."
            )
        return images.to(self.device)

    def _next_fake_volumes(self) -> torch.Tensor:
        try:
            batch = next(self.fake_iterator)
        except StopIteration:
            self.fake_iterator = iter(self.fake_dataloader)
            try:
                batch = next(self.fake_iterator)
            except StopIteration as exc:
                raise ValueError("fake dataloader cannot provide latent volumes.") from exc
        if not isinstance(batch, torch.Tensor) or batch.ndim != 5:
            raise TypeError("fake batch must have shape [B, C, D, H, W].")
        return batch

    def _save(self) -> None:
        if not self.is_main_process or self.step % self.save_every != 0:
            return
        checkpoint = {
            "step": self.step,
            "generator": unwrap_model(self.generator).state_dict(),
            "critic": unwrap_model(self.critic).state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        step_dir = self.weight_dir / str(self.step)
        write_checkpoint(checkpoint, step_dir / "model.pt")
        write_checkpoint(checkpoint, self.last_weight_dir / "model.pt")
