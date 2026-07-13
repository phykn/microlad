# Conditioned MicroLad

MicroLad-style 2D-to-3D microstructure generation.

This repo trains on 2D grayscale or phase image patches, then generates a 3D candidate. During inference, observed 2D slices guide specified positions through a soft anchor loss, and target statistics from a slice bundle can guide SDS refinement.

The active implementation is in `src`, and maintained interactive examples are in `notebooks`. Legacy reference folders are not part of the maintained workflow.
A small sample image is checked in under `data/` so the default config can build a dataset without external files.
Real training datasets and model checkpoints are not checked in.

The source tree has four ownership layers:

- `common`: shared image, neural, and tensor utilities
- `modeling`: phase representation, VAE, and diffusion models
- `pipelines`: data, training, reconstruction, guidance, and scale-up workflows
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
- The default `data.data_dir` points at the checked-in sample image.
- For real training, put images outside the repo or in a folder ignored by your local git exclude, then set `data.data_dir` in the config.
- Train VAE first, then train diffusion.
- For local diffusion training, set `output.vae_run_dir` in `config/diffusion.yaml` to the VAE run folder before launching `run_train_diffusion.py`.
- VAE reconstruction is categorical: the decoder emits `[B, num_phases, H, W]`
  logits and the VAE loss is `CE(logits, phase_index) + beta * KL`.

## Test

Run the full suite:

```sh
python -m pytest -q
```

For a bounded training-entrypoint check:

```sh
python -m pytest tests/app/runtime/test_run_train_vae.py tests/app/runtime/test_run_train_diffusion.py -q
```

## Train

The default training configs are full-length examples.

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

## Notebooks

The notebooks follow the maintained workflow in order:

- `00_dataset.ipynb`: load and inspect phase-index batches
- `01_vae.ipynb`: compare VAE inputs and reconstructions
- `02_diffusion.ipynb`: sample and decode diffusion latents
- `03_predict.ipynb`: run anchored prediction and inspect descriptor targets
- `04_scale_up.ipynb`: generate and inspect a larger volume

`RUN_DIR = None` selects the latest compatible run. Set it to a run path when a specific checkpoint is required. Dataset tensors, decoded VAE values, and descriptor inputs use phase indices from `0` to `num_phases - 1`.

## Predict

Load a trained run folder and call `predict`:

```python
from src.app.api import AnchorSlice, PredictOptions
from src.app.runtime import load_predictor, load_slicegan_config

predictor = load_predictor("run/20260628-xxxxxx", device="cuda")

options = PredictOptions(
    num_phases=3,
    phase_fractions=(0.28, 0.12, 0.60),
    slicegan=load_slicegan_config("config/slicegan.yaml"),
)

anchor = AnchorSlice(image=anchor_image, axis=0, index=32)
volume, stats = predictor.predict(options, anchors=[anchor])
```

`config/slicegan.yaml` groups training, anchor conditioning, and large-volume rendering settings. Normal prediction code only selects that config instead of listing every optimization parameter.

Anchors are full 2D slices assigned to a volume axis and index. Conditional SliceGAN uses them as constraints; generated values are not forcibly overwritten.

For scale-up, pass a larger `volume_size` or provide larger anchor slices. SDS descriptor targets live under `TargetConfig`, for example `targets=TargetConfig(vf_weight=1.0, tpc_weight=1.0)`.

## Reference

- Original GitHub: [KangHyunL/microlad](https://github.com/KangHyunL/microlad)
- Paper: [MicroLad](https://arxiv.org/abs/2508.20138)
