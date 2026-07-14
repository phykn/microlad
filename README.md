# Conditioned MicroLad

MicroLad-style 2D-to-3D microstructure generation.

This repo trains on 2D grayscale or phase image patches, then generates a 3D candidate. During inference, observed 2D slices guide specified positions through a soft anchor loss, and target statistics from a slice bundle can guide SDS refinement.

The active implementation is in `src`, and maintained interactive examples are in `notebooks`. Legacy reference folders are not part of the maintained workflow.
A small sample image is checked in under `data/` so the default config can build a dataset without external files.
Real training datasets and model checkpoints are not checked in.

The source tree has four ownership layers:

- `common`: shared image, neural, and tensor utilities
- `modeling`: phase representation, VAE, and diffusion models
- `pipeline`: shared `data`, model `train`, and volume `predict` workflows
- `app`: prediction API, configuration, loading, and object construction

## Install

```sh
python -m pip install -r requirements.txt
```

The requirements include training, tests, and the libraries imported by the notebooks. Select this environment as the notebook kernel.

Run training and prediction examples from the repository root. For scripts or
interactive sessions launched elsewhere, add the checkout root to `PYTHONPATH` so imports
such as `from src.app.runtime import load_predictor` resolve.

## Config

- VAE config: `config/vae.yaml`
- Diffusion config: `config/diffusion.yaml`
- GAN config: `config/gan.yaml`
- The default `data.data_dir` points at the checked-in sample image.
- For real training, put images outside the repo or in a folder ignored by your local git exclude, then set `data.data_dir` in the config.
- Train the VAE first. Diffusion and the 2D latent WGAN can then be trained independently.
- For local diffusion training, set `output.vae_run_dir` in `config/diffusion.yaml` to the VAE run folder before launching `run_train_diffusion.py`.
- Before GAN training, set `output.vae_run_dir` in `config/gan.yaml` to the VAE run folder.
- Set the VAE, diffusion, and GAN run folders under `models` in `config/predict.yaml` before prediction.
- VAE reconstruction is categorical: the decoder emits `[B, num_phases, H, W]`
  logits and the VAE loss is phase-balanced `CE(logits, phase_index) + beta * KL`.
  `loss.phase_balance=0` restores ordinary CE; the default `0.35` moderately
  raises the contribution of a rare phase without equalizing every phase fully.
- Diffusion checkpoints store an exponential moving average of the online model.
  `training.ema_decay` controls the update and does not change the fixed step or
  checkpoint schedule.
- Final Refine is controlled only by `refine.enabled`; when enabled it runs once.

## Test

Run the full suite:

```sh
python -m pytest -q
```

For a bounded training-entrypoint check:

```sh
python -m pytest tests/app/runtime/test_run_train_vae.py tests/app/runtime/test_run_train_diffusion.py tests/app/runtime/test_run_train_gan.py -q
```

## Train

The default training configs are full-length examples.

```sh
python run_train_vae.py
python run_train_diffusion.py
python run_train_gan.py
```

DDP:

```sh
torchrun --nproc_per_node 4 run_train_vae.py
torchrun --nproc_per_node 4 run_train_diffusion.py
torchrun --nproc_per_node 4 run_train_gan.py
```

Training writes to `run/<timestamp>` by default.

A VAE run contains:

```text
run/<timestamp>/
  vae.yaml
  weight/
    vae/last/model.pt
```

A diffusion run copies the VAE config and last VAE checkpoint, then adds diffusion state:

```text
run/<timestamp>/
  vae.yaml
  diffusion.yaml
  weight/
    vae/last/model.pt
    diffusion/last/model.pt
```

A GAN run copies the frozen VAE and adds the 2D generator and critic state:

```text
run/<timestamp>/
  vae.yaml
  gan.yaml
  weight/
    vae/last/model.pt
    gan/last/model.pt
```

## Data Shape

- 2D images: `H x W`
- Dataset output: `[1, H, W]`, float phase indices from `0` to
  `num_phases - 1`
- VAE decoder training output: `[num_phases, H, W]` logits
- VAE latent: `[C, 16, 16]` by default
- If `segment: true`, dataset inputs are loaded as grayscale `uint8`,
  then segmented into phase indices from `0` to `num_phases - 1`.
- If `segment: false`, dataset inputs must already be 2D phase label images
  with integer values from `0` to `num_phases - 1`.
- Diffusion runs inherit VAE preprocessing fields from `output.vae_run_dir`
  (`crop_size`, `size`, `segment`, `num_phases`) so the latent dataset matches
  the frozen VAE.

## Notebooks

The notebooks follow the maintained workflow in order:

- `00_dataset.ipynb`: load and inspect phase-index batches
- `01_vae.ipynb`: compare VAE inputs and reconstructions
- `02_diffusion.ipynb`: sample and decode diffusion latents
- `03_gan.ipynb`: sample the latent WGAN and inspect critic scores
- `04_predict.ipynb`: run anchored prediction and inspect descriptor targets
- `05_scale_up.ipynb`: generate and inspect a larger volume

The prediction notebooks read all model folders from `config/predict.yaml`. Dataset tensors, decoded VAE values, and descriptor inputs use phase indices from `0` to `num_phases - 1`.

## Predict

Load a trained run folder and call `predict`:

```python
from src.app.api import AnchorSlice, PredictOptions
from src.app.runtime import load_predict_config, load_predictor

model_runs, predict_config = load_predict_config("config/predict.yaml")
predictor = load_predictor(**model_runs, device="cuda")

options = PredictOptions(
    num_phases=3,
    **predict_config,
)

anchor = AnchorSlice(image=anchor_image, axis=0, index=32)
volume, stats = predictor.predict(
    options,
    anchors=[anchor],
)
```

Train the unconditional 2D latent WGAN with `run_train_gan.py` after VAE
training. Its run copies the VAE checkpoint and adds the GAN checkpoint. Prediction
loads VAE, diffusion, and GAN runs independently from `config/predict.yaml` and uses
only the frozen critic; the WGAN generator is retained for `03_gan.ipynb` evaluation.
`config/predict.yaml` controls the critic guidance weight along with the L-MPDD prior,
latent refinement, optional one-pass Refine, and scale-up settings.

`joint.decode_batch_size` controls decoder memory use. Keep a positive batch size
for chunked decoding with gradient checkpointing, or set it to `null` to decode
each axis in one batch on a large-memory GPU.

Prediction progress is shown by default for L-MPDD sampling, Joint guidance, and
scale-up guidance. Set `progress: false` in the prediction config to hide every
progress bar. GAN training reports generator loss, critic loss, margin, gradient
penalty. Joint reports total loss and active anchor, critic,
and global fraction losses.

Anchors are full 2D slices assigned to a volume axis and index. They are soft
decoded constraints, so target labels are not forcibly copied into the result.
Multiple-axis anchor intersections must request compatible categorical labels.

For scale-up, pass a larger `volume_size`; the `scale` section controls tiled
L-MPDD sampling and one shared 3D latent residual. Guidance samples balanced
latent crops from all three axes, so it never optimizes or overwrites scalar phase
label slices. The final scale-optimized latent is decoded directly. Scale anchors use
the same tiled tri-axis probability consensus
as final decoding, sampled over anchors and image regions during optimization without
copying target labels. `scale.decode_batch_size` limits tiled sampling, anchor loss,
decoding, and refinement memory; set it to `null` to process each stage in one batch
on a large-memory GPU. Both modes use the same decoder meaning. Reference descriptor
targets live under `TargetConfig`, for
example `targets=TargetConfig(slice_fraction_weight=1.0, tpc_weight=1.0)`.
Explicit `phase_fractions` use `global_fraction_weight` and do not constrain
every slice unless `slice_fraction_weight` is also enabled.
Set `phase_fractions: null` to derive the target from `target_images`, or from the
anchors when no separate target bundle is provided. An explicit list conditions on a
requested global composition. `target_images` remain optional morphology references
for final diagnostics and are not merged into the anchor constraints.

## Reference

- Original GitHub: [KangHyunL/microlad](https://github.com/KangHyunL/microlad)
- Paper: [MicroLad](https://arxiv.org/abs/2508.20138)
