# MicroLAD

MicroLAD is an experimental image-space Multi-Plane Denoising Diffusion
(MPDD) implementation for generating 3D categorical microstructures from 2D
label images. It does not require 3D training volumes or a VAE.

This codebase differs from the original MicroLad paper and repository, which
use latent diffusion and score distillation sampling (SDS). The current
implementation focuses on direct categorical diffusion, multi-plane sampling,
phase-fraction conditioning, soft slice anchors, and tiled scale-up.

## Quick start

```powershell
python -m pip install -r requirements.txt
python run_train.py
```

Training is configured in `config/model.yaml`. Input images should contain
integer phase labels from `0` to `K - 1`; intensity images can instead be
segmented by enabling `segment`. Checkpoints are written to
`run/<timestamp>/weight/mpdd/`.

Generation is configured in `config/predict.yaml`.

```python
from src.misc import load_config
from src.predict import MPDDOptions, load_predictor

config = load_config("config/predict.yaml")
predictor = load_predictor(config.pop("run_dir"))
options = MPDDOptions(**config)
volume, stats = predictor.predict(options)
```

`volume` is a `uint8` tensor with shape `[D, H, W]`. Runnable examples in
`notebooks/` cover data inspection, model checks, 2D diffusion, 3D sampling,
anchors, and scale-up.

## Method

### Problem

Let each 2D training image be a categorical field

$$
y \in \{0, \ldots, K-1\}^{H \times W}.
$$

The goal is to sample a 3D categorical volume

$$
V \in \{0, \ldots, K-1\}^{D \times H \times W}
$$

whose slices resemble the training distribution. This is not recovery of a
unique original 3D object. Multiple 3D structures can share similar 2D slice
statistics, so the result is one compatible sample rather than a deterministic
reconstruction.

### Image-space categorical diffusion

Each label image is converted to centered one-hot channels:

$$
x_0 = 2\,\operatorname{onehot}(y)-1.
$$

The forward DDPM process adds Gaussian noise:

$$
x_t =
\sqrt{\bar{\alpha}_t}\,x_0 +
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
$$

A 2D U-Net predicts the injected noise,
$\epsilon_\theta(x_t,t,c)$, where $c$ is an optional phase-fraction
condition. Training minimizes mean squared error between the sampled and
predicted noise. After sampling, the phase with the largest channel value is
selected at every voxel:

$$
V = \operatorname*{argmax}_k x^{(k)}.
$$

Working directly in image space lets the denoiser observe phase boundaries and
particle geometry at the target resolution. It also keeps the training input,
diffusion state, and decoded output in the same representation.

### Multi-plane sampling

Sampling starts from one 3D Gaussian noise tensor `X_T`. At each reverse step,
the sampler:

1. chooses one of the three spatial axes;
2. views the volume as a batch of 2D planes;
3. denoises those planes with the trained 2D model; and
4. merges them back into the same 3D state.

For a slice operator `S_a` along axis `a`, one update can be written
conceptually as

$$
X_{t'}
=
S_a^{-1}
\left(
\mathcal{D}_{\theta,t\rightarrow t'}
\left(
S_a(X_t)
\right)
\right),
$$

where `t' = t - 1` for DDPM and may skip steps for DDIM. The axis rotates
across successive reverse steps. Because every axis updates the same tensor,
changes made from one view become input to later views.

This procedure is best understood as alternating refinement with learned 2D
slice priors. It encourages compatibility across views, but it does not
mathematically identify or guarantee the true 3D joint distribution.

### Harmonization and DDIM

One plane update may not move the current volume close enough to the learned 2D
distribution. Harmonization therefore repeats denoising and re-noising at the
same schedule step before moving on. The parameter `harmonization_steps`
controls this repetition.

A larger value uses more denoiser evaluations and can strengthen the selected
plane prior, but quality is not guaranteed to improve monotonically. DDIM
reduces the number of reverse steps by connecting selected cumulative-alpha
states deterministically. `ddim_steps` and `harmonization_steps` therefore
trade sampling cost against the amount of plane-wise refinement.

## Conditioning

### Phase fractions

The phase-fraction vector

$$
c \in \Delta^{K-1}
$$

is embedded and added to the timestep embedding. During training, some
conditions are replaced by a learned null condition. This enables
classifier-free guidance at inference:

$$
\hat{\epsilon}
=
\epsilon_\theta(x_t,t,\varnothing)
+
s\left[
\epsilon_\theta(x_t,t,c)
-
\epsilon_\theta(x_t,t,\varnothing)
\right].
$$

The condition changes the denoising direction but is not a hard volume
constraint. The final fractions can differ from the target, and the response
depends on the training distribution, volume size, anchors, and guidance
scale.

### Soft slice anchors

A labeled 2D observation can guide one plane of the sampled volume. The anchor
is converted to centered one-hot channels and diffused to the current noise
level:

$$
x_t^A =
\sqrt{\bar{\alpha}_t}\,x_0^A +
\sqrt{1-\bar{\alpha}_t}\,\epsilon_A.
$$

The sampler reuses one noise realization `epsilon_A` for each anchor
throughout the trajectory and blends the noisy anchor into its 3D mask during
the high-noise portion of sampling. The anchor is released during the final
low-noise portion so the denoiser can redraw its boundary with the surrounding
structure.

Anchors are therefore soft guidance, not exact final constraints. Multiple
anchors can be combined, while duplicate planes and conflicting intersections
are rejected before sampling.

## Tiled scale-up

The denoiser is trained on fixed-size 2D images. For a larger volume, each
plane is split into overlapping training-size tiles. Predicted noise is merged
with a smooth window:

$$
\hat{\epsilon}(p)
=
\frac{\sum_i w_i(p)\,\hat{\epsilon}_i(p)}
{\sum_i w_i(p)}.
$$

Overlap reduces discontinuities at tile boundaries and allows larger volumes
without retraining the network. This assumes that the microstructure is
locally stationary. Correlations, connected structures, or directional
patterns longer than a tile are not explicitly modeled by overlap alone.

## Scope and limitations

1. A 2D slice distribution does not uniquely determine a 3D joint
   distribution.
2. Realistic slices do not guarantee correct 3D connectivity, topology, or
   physical properties.
3. The same 2D denoiser is applied along all three axes, so axis-specific
   anisotropy is not represented explicitly.
4. Phase fractions and anchors are soft conditions rather than exact
   constraints.
5. Tiled scale-up preserves local behavior more directly than long-range
   structure.
6. Final argmax conversion is discontinuous and can change boundaries when
   channel scores are close.

The working hypothesis is that, when important microstructure information is
well represented by 2D slice statistics, alternating one image-space denoiser
over a shared 3D diffusion state can produce compatible 3D categorical
samples. This hypothesis must be evaluated across datasets rather than inferred
from one visually successful example.

Useful 2D checks include phase fractions, particle-size distributions,
circularity, interface length, and two-point correlations. Useful 3D checks
include connected components, interfacial area, directional correlations,
percolation, and property-based metrics. These evaluation metrics are not
implemented by the current core package.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## References

- [MicroLad paper: latent diffusion and score distillation](https://arxiv.org/abs/2508.20138)
- [Original MicroLad repository](https://github.com/KangHyunL/microlad)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
