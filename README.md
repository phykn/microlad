# Conditioned MicroLad

MicroLad-style 2D-to-3D microstructure generation.

This repo trains on 2D grayscale or phase image patches, then generates a 3D candidate. During inference, observed 2D slices can be fixed at specified positions, and target statistics from a slice bundle can guide SDS refinement.

The active implementation is in `src`. Legacy reference folders and removed interactive examples are not part of the maintained workflow.
A small sample image is checked in under `data/` so the default config can build a dataset without external files.
Real training datasets and model checkpoints are not checked in.

The source tree is grouped by responsibility:

- `phases`, `vae`, `diffusion`: mathematical representation and learned models
- `reconstruction`, `guidance`, `scaling`: 3D generation, controlled objectives, and large-volume tiling
- `api`, `runtime`: prediction facade, configuration, loading, and object construction
- `data`, `io`, `training`: external data boundaries and training lifecycle

## Install

```sh
python -m pip install -r requirements.txt
```

Run training and prediction examples from the repository root. For scripts or
interactive sessions launched elsewhere, add the checkout root to `PYTHONPATH` so imports
such as `from src.runtime import load_predictor` resolve.

## Config

- VAE config: `config/vae.yaml`
- Diffusion config: `config/diffusion.yaml`
- The default `data.data_dir` points at the checked-in sample image.
- For real training, put images outside the repo or in a folder ignored by your local git exclude, then set `data.data_dir` in the config.
- Train VAE first, then train diffusion.
- For local diffusion training, set `output.vae_run_dir` in `config/diffusion.yaml` to the VAE run folder before launching `run_train_diffusion.py`.
- VAE reconstruction is categorical: the decoder emits `[B, num_phases, H, W]`
  logits and the VAE loss is `CE(logits, phase_index) + beta * KL`.

## Train

The default training configs are full-length examples. For a bounded smoke
check, run the focused entrypoint tests instead:

```sh
python -m pytest tests/test_run_train_vae.py tests/test_run_train_diffusion.py -q
```

```sh
python run_train_vae.py
python run_train_diffusion.py
```

DDP:

```sh
torchrun --nproc_per_node 4 run_train_vae.py
torchrun --nproc_per_node 4 run_train_diffusion.py
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

## Predict

Load a trained run folder and call `predict`:

```python
from src.api import AnchorSlice, PredictOptions
from src.runtime import load_predictor

predictor = load_predictor("run/20260628-xxxxxx", device="cuda")

options = PredictOptions(
    num_phases=3,
    sds_steps=0,
    refine_steps=0,
)

volume, stats = predictor.predict(options)
```

Anchors are full 2D slices fixed at a volume axis and index:

```python
anchor = AnchorSlice(image=anchor_image, axis=0, index=32)
volume, stats = predictor.predict(options, anchors=[anchor])
```

For scale-up, pass a larger `volume_size` or provide larger anchor slices. Target-image bundles can be passed with SDS target weights such as `vf_weight`, `tpc_weight`, `sa_weight`, or `diffusivity_weight`.

## Reference

- Original GitHub: [KangHyunL/microlad](https://github.com/KangHyunL/microlad)
- Paper: [MicroLad](https://arxiv.org/abs/2508.20138)
