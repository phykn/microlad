# MicroLad MECE Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 수학 객체와 파일 책임이 일치하는 MECE package 구조로 전체 프로젝트를 재구성하고, scale overlap이 local gradient를 중복 가중하는 오류를 수정한다.

**Architecture:** `models/loss/predict` 기술 계층을 없애고 `phases`, `vae`, `diffusion`, `reconstruction`, `guidance`, `scaling`, `api`, `runtime` 도메인 package로 이동한다. 이동 단계는 기존 동작을 보존하며, overlap normalization만 별도 TDD 단계에서 변경한다.

**Tech Stack:** Python 3.13, PyTorch 2.11, pytest, YAML

## Global Constraints

- 공개 import, 설정 키, checkpoint 형식의 하위 호환성은 요구하지 않는다.
- 사용자 추가 스케일업은 핵심 기능으로 보존한다.
- phase RGB 입력은 첫 번째 channel을 사용한다.
- 검증되지 않은 KL, latent scaling, SDS weighting 변경은 하지 않는다.
- 파일은 단일 책임을 가지며 같은 이유로 변경되는 파일은 같은 package에 둔다.
- `utils.py`, `common.py`, `types.py`라는 새 파일 이름을 만들지 않는다.

---

### Task 1: Phase, neural primitive, VAE package

**Files:**
- Create: `src/phases/__init__.py`
- Move: `src/loss/phase.py` → `src/phases/representation.py`
- Move: `src/segment/multi_otsu.py` → `src/phases/segmentation.py`
- Move: `src/predict/postprocess.py` → `src/phases/quantization.py`
- Create: `src/neural/__init__.py`
- Move: `src/models/norm.py` → `src/neural/normalization.py`
- Move: `src/models/shape.py` → `src/neural/spatial.py`
- Create: `src/vae/__init__.py`
- Move: `src/models/vae.py` → `src/vae/model.py`
- Merge: `src/loss/kl.py`, `src/loss/vae.py` → `src/vae/objective.py`
- Modify: all imports and phase/VAE tests

**Interfaces:**
- Produces: `PatchVAE`, `vae_loss`, `kl_divergence`, phase representation and quantization functions
- Consumes: reusable `neural` normalization and spatial shape helpers

- [ ] **Step 1: 새 import를 요구하도록 집중 테스트 import 변경**

```python
from src.phases import logits_to_relaxed_labels, phase_cross_entropy
from src.vae import PatchVAE, kl_divergence, vae_loss
```

Run: `python -m pytest tests/test_vae.py tests/test_vae_loss.py -q`

Expected: FAIL with missing `src.phases` or `src.vae`.

- [ ] **Step 2: 파일을 이동하고 명확한 phase API 제공**

`src/phases/representation.py`에서 기존 `logits_to_phase_values`를 다음 이름으로 바꾼다.

```python
def logits_to_probabilities(logits: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_phase_logits(logits, num_phases)
    return torch.softmax(logits, dim=1)


def logits_to_relaxed_labels(logits: torch.Tensor, num_phases: int) -> torch.Tensor:
    probabilities = logits_to_probabilities(logits, num_phases)
    levels = phase_levels(num_phases, device=logits.device, dtype=logits.dtype)
    return (probabilities * levels.view(1, num_phases, 1, 1)).sum(dim=1, keepdim=True)


def logits_to_labels(logits: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_phase_logits(logits, num_phases)
    return logits.argmax(dim=1, keepdim=True)
```

`PatchVAE`는 `decode_probabilities`, `decode_relaxed_labels`를 제공하고 기존 내부 호출은 의도를 드러내는 새 이름을 사용한다.

- [ ] **Step 3: VAE objective를 한 파일로 합치고 import 수정**

`src/vae/objective.py`는 `kl_divergence`, `vae_loss`, `VAELoss`만 소유한다. reduction은 현재 element mean을 유지하고 docstring에 명시한다.

- [ ] **Step 4: 집중 테스트**

Run: `python -m pytest tests/test_vae.py tests/test_vae_loss.py tests/test_segmentation.py tests/test_predict_postprocess.py tests/math_audit/test_core_equations.py -q`

Expected: all pass.

- [ ] **Step 5: 커밋**

```bash
git add src/phases src/neural src/vae tests
git commit -m "refactor: group phase and VAE responsibilities"
```

---

### Task 2: Diffusion package

**Files:**
- Create: `src/diffusion/__init__.py`
- Move: `src/models/ddpm.py` → `src/diffusion/process.py`
- Move: `src/models/unet.py` → `src/diffusion/model.py`
- Move: `src/loss/diffusion.py` → `src/diffusion/objective.py`
- Move: `src/predict/sampler/diffusion.py` → `src/diffusion/sampler.py`
- Modify: diffusion imports, tests, factories

**Interfaces:**
- Produces: `DDPMProcess`, `TimeUNet`, `diffusion_loss`, `DiffusionSampler`
- Consumes: `src.neural`

- [ ] **Step 1: process 이름을 요구하는 테스트 변경**

```python
from src.diffusion import DDPMProcess, DiffusionSampler, TimeUNet
```

Run: `python -m pytest tests/test_diffusion_models.py tests/test_diffusion_loss.py tests/test_predict_sampler.py -q`

Expected: FAIL because `src.diffusion` does not exist.

- [ ] **Step 2: 파일 이동과 class rename**

`DDPM`을 `DDPMProcess`로 바꾸고 type annotation과 factory 이름을 `build_diffusion_process`로 바꾼다. schedule 식은 변경하지 않는다.

- [ ] **Step 3: 집중 테스트**

Run: `python -m pytest tests/test_diffusion_models.py tests/test_diffusion_loss.py tests/test_predict_sampler.py tests/math_audit/test_core_equations.py -q`

Expected: all pass.

- [ ] **Step 4: 커밋**

```bash
git add src/diffusion src/neural src tests
git commit -m "refactor: group diffusion process and sampler"
```

---

### Task 3: Reconstruction, guidance, scaling, API package 이동

**Files:**
- Move: `src/predict/slices.py`, `volume.py`, `refine.py` → `src/reconstruction/`
- Move: `src/predict/sds/` → `src/guidance/`
- Move: `src/predict/scale/` and `src/predict/blend.py` → `src/scaling/`
- Move: `src/predict/anchor/` and `src/predict/targets.py` → `src/guidance/conditioning/`
- Move: `src/predict/predictor.py`, `types.py` → `src/api/`
- Modify: all imports and tests

**Interfaces:**
- Produces: reconstruction geometry, guidance objectives, scale adapters, public `Predictor` and `PredictOptions`
- Consumes: phase, VAE and diffusion packages

- [ ] **Step 1: 새 public import로 API 테스트 변경**

```python
from src.api import AnchorSlice, Predictor, PredictOptions
from src.guidance.descriptors import volume_fraction_loss
from src.reconstruction.slices import extract_slice
from src.scaling.tiles import tile_grid
```

Run: `python -m pytest tests/test_predict_predictor.py tests/test_predict_slices.py tests/test_predict_scale_tiles.py tests/test_predict_sds_vf.py -q`

Expected: FAIL with missing packages.

- [ ] **Step 2: reconstruction과 scaling 이동**

이름을 다음처럼 정리한다.

```text
refine.py -> refinement.py
denoise.py -> denoising.py
decode.py -> decoding.py
sampler.py -> sampling.py
blend.py -> blending.py
```

- [ ] **Step 3: guidance 내부를 역할별 subpackage로 이동**

```text
vf.py -> descriptors/volume_fraction.py
tpc.py -> descriptors/two_point_correlation.py
sa.py -> descriptors/surface_area.py
diffusivity.py -> physics/diffusivity.py
core.py -> prior.py
optimize.py -> optimization.py
targets.py -> conditioning/targets.py
anchor/*.py -> conditioning/anchor_*.py
```

- [ ] **Step 4: public API 이동과 import 전환**

`src/api/__init__.py`는 `AnchorSlice`, `PredictOptions`, `Predictor`만 export한다. 내부 package의 private helper는 export하지 않는다.

- [ ] **Step 5: 집중 테스트**

Run: `python -m pytest tests/test_predict_slices.py tests/test_predict_volume.py tests/test_predict_refine.py tests/test_predict_anchor.py tests/test_predict_sds_core.py tests/test_predict_sds_objective.py tests/test_predict_scale_tiles.py tests/test_predict_scale_decode_refine.py tests/test_predict_predictor.py -q`

Expected: all pass.

- [ ] **Step 6: 커밋**

```bash
git add src/reconstruction src/guidance src/scaling src/api src tests
git commit -m "refactor: separate reconstruction guidance and scaling"
```

---

### Task 4: Scale overlap gradient normalization

**Files:**
- Modify: `src/scaling/tiles.py`
- Modify: `src/scaling/optimization.py`
- Modify: `src/guidance/prior.py`
- Modify: `src/guidance/conditioning/anchor_objective.py`
- Test: `tests/math_audit/test_geometry_and_scale.py`
- Test: `tests/test_predict_scale_sds.py`

**Interfaces:**
- Produces: `normalized_tile_weights(...)`, optional `spatial_weight` for SDS and anchor objectives
- Consumes: tile grid and tensor spatial shapes

- [ ] **Step 1: 원하는 균일 gradient 테스트로 변경**

```python
def test_local_tile_loss_gradient_is_normalized_by_tile_coverage():
    # same 4x4 / 3x3 overlap setup as the audit characterization
    corner = gradient[0, 0].abs()
    center = gradient[1, 1].abs()
    assert torch.allclose(center, corner, atol=1e-6, rtol=1e-6)
```

Run: `python -m pytest tests/math_audit/test_geometry_and_scale.py::test_local_tile_loss_gradient_is_normalized_by_tile_coverage -q`

Expected: FAIL; current center/corner ratio is 4.

- [ ] **Step 2: normalized ownership map 구현**

```python
def normalized_tile_weights(height, width, *, tile_size, overlap, device, dtype):
    window = blend_window(tile_size, tile_size, device=device, dtype=dtype)
    placements = list(tile_grid(height, width, tile_size=tile_size, overlap=overlap))
    total = torch.zeros(height, width, device=device, dtype=dtype)
    for row, col in placements:
        total[row:row + tile_size, col:col + tile_size] += window
    return [
        (row, col, window / total[row:row + tile_size, col:col + tile_size])
        for row, col in placements
    ]
```

- [ ] **Step 3: anchor와 SDS loss에 spatial weight 적용**

SDS surrogate와 pixelwise anchor loss는 `sum(loss * weight) / sum(weight)` reduction을 사용한다. stitched output은 동일 ownership weight를 사용한다. descriptor whole-slice는 stitched 결과에서 한 번 계산한다.

- [ ] **Step 4: 집중 테스트**

Run: `python -m pytest tests/math_audit/test_geometry_and_scale.py tests/test_predict_scale_sds.py tests/test_predict_sds_anchor.py tests/test_predict_sds_core.py -q`

Expected: all pass and center/corner gradients match.

- [ ] **Step 5: 커밋**

```bash
git add src/scaling src/guidance tests
git commit -m "fix: normalize overlapping scale guidance"
```

---

### Task 5: Training, data, IO package 이름 정리

**Files:**
- Move: `src/train/` → `src/training/`
- Move: `src/data/patch_dataset.py` → `src/data/dataset.py`
- Consolidate: crop/resize/augment exports in `src/data/transforms.py`
- Move: `src/io/image.py` → `src/io/images.py`
- Modify: imports, tests, run scripts

**Interfaces:**
- Produces: training loops, dataset, transforms, external image loading
- Consumes: VAE, diffusion, runtime factories

- [ ] **Step 1: 새 import로 관련 테스트 변경 후 RED 확인**

Run: `python -m pytest tests/test_dataset.py tests/test_data_augment.py tests/test_image_loader.py tests/test_vae_trainer.py tests/test_diffusion_trainer.py -q`

Expected: FAIL after test imports point to `src.training` and `src.io.images`.

- [ ] **Step 2: 파일 이동과 import 수정**

`training/runtime.py`는 logging, checkpoint, gradient norm과 run directory lifecycle을 소유한다. dataset transforms는 데이터 tensor 변환만 소유한다.

- [ ] **Step 3: 집중 테스트와 entrypoint smoke**

Run: `python -m pytest tests/test_dataset.py tests/test_data_augment.py tests/test_image_loader.py tests/test_train_utils.py tests/test_vae_trainer.py tests/test_diffusion_trainer.py tests/test_run_train_vae.py tests/test_run_train_diffusion.py -q`

Expected: all pass.

- [ ] **Step 4: 커밋**

```bash
git add src/data src/io src/training run_train_vae.py run_train_diffusion.py tests
git commit -m "refactor: clarify data IO and training packages"
```

---

### Task 6: Runtime build 분리와 public loading 경로

**Files:**
- Delete: `src/build.py`
- Create: `src/runtime/config.py`
- Create: `src/runtime/distributed.py`
- Create: `src/runtime/factories.py`
- Create: `src/runtime/loading.py`
- Create: `src/runtime/__init__.py`
- Modify: run scripts, tests, README, notebooks

**Interfaces:**
- Produces: config IO, device/distributed lifecycle, object factories, `load_predictor`
- Consumes: all domain packages but owns no math

- [ ] **Step 1: 새 runtime import로 build 테스트 변경**

```python
from src.runtime import load_config_defaults, load_predictor
from src.runtime.factories import build_dataset, build_vae_trainer
```

Run: `python -m pytest tests/test_build.py -q`

Expected: FAIL because runtime modules are missing.

- [ ] **Step 2: `build.py` 함수를 네 책임으로 이동**

```text
config.py: flatten/load/save/copy/fill defaults
distributed.py: setup/cleanup/wrap distributed
factories.py: dataset/model/process/optimizer/trainer factories
loading.py: checkpoint validation and predictor loading
```

- [ ] **Step 3: public entrypoint와 문서 수정**

README와 notebook 예제는 `from src.runtime import load_predictor`를 사용한다. maintained run scripts는 `runtime.config`, `runtime.distributed`, `runtime.factories`만 import한다.

- [ ] **Step 4: 집중 테스트**

Run: `python -m pytest tests/test_build.py tests/test_run_train_vae.py tests/test_run_train_diffusion.py tests/test_predict_predictor.py -q`

Expected: all pass.

- [ ] **Step 5: 커밋**

```bash
git add src/runtime src/build.py run_train_vae.py run_train_diffusion.py README.md notebooks tests
git commit -m "refactor: split runtime configuration and factories"
```

---

### Task 7: Hotspot 분리와 명명 정리

**Files:**
- Split: `src/guidance/optimization.py`
- Split: `src/scaling/optimization.py`
- Split: `src/api/predictor.py`
- Modify: corresponding imports and tests

**Interfaces:**
- Produces: focused optimization loop, objective evaluator, validation contract, API orchestration
- Consumes: established package public interfaces

- [ ] **Step 1: 각 hotspot의 함수 그룹을 별도 파일로 이동**

```text
guidance/optimization.py: public slice/volume loop only
guidance/evaluation.py: objective evaluation only
guidance/validation.py: optimization input contracts only
scaling/optimization.py: public large-volume loop only
scaling/local_objective.py: tiled objective and decoding only
scaling/validation.py: scale optimization contracts only
api/predictor.py: predict orchestration only
api/preparation.py: target/anchor/options preparation only
```

- [ ] **Step 2: 파일 크기와 이름 검사**

Run: `Get-ChildItem src -Recurse -Filter *.py | ForEach-Object { [PSCustomObject]@{Lines=(Get-Content $_).Count; Path=$_.FullName} } | Sort-Object Lines -Descending | Select-Object -First 10`

Expected: orchestration 파일이 objective/validation을 함께 소유하지 않는다. 500줄을 넘는 파일이 있으면 한 책임인지 재검토한다.

- [ ] **Step 3: 전체 prediction 집중 테스트**

Run: `python -m pytest tests/test_predict_*.py -q`

Expected: all pass.

- [ ] **Step 4: 커밋**

```bash
git add src/api src/guidance src/scaling tests
git commit -m "refactor: split prediction orchestration hotspots"
```

---

### Task 8: 최종 검증과 문서 정합성

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md` only if maintained entrypoint guidance changed
- Modify: relevant notebooks
- Verify: all source and tests

**Interfaces:**
- Produces: self-consistent maintained project surface

- [ ] **Step 1: 제거된 import 검색**

Run: `rg -n "src\.(build|models|loss|predict|train|segment)" src tests run_train_vae.py run_train_diffusion.py README.md notebooks`

Expected: no matches.

- [ ] **Step 2: syntax와 import compile**

Run: `python -m compileall -q src tests run_train_vae.py run_train_diffusion.py`

Expected: exit 0.

- [ ] **Step 3: 전체 테스트**

Run: `python -m pytest -q`

Expected: all tests and subtests pass.

- [ ] **Step 4: 감사 property 재검증**

Run: `python -m pytest tests/math_audit -q`

Expected: all pass, including normalized overlap gradient.

- [ ] **Step 5: worktree와 diff 검토**

Run: `git status --short --branch`

Expected: only intended final documentation changes, or clean after commit.

Run: `git diff --check HEAD~1..HEAD`

Expected: no whitespace errors.

- [ ] **Step 6: 최종 커밋**

```bash
git add src tests run_train_vae.py run_train_diffusion.py README.md notebooks AGENTS.md
git commit -m "docs: align project with MECE package structure"
```
