# MicroLad Mathematical Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 논문 수식, 현재 코드, 작은 합성 실험을 연결해 MicroLad 전체 수학 감사 보고서와 이후 리팩터링의 검증 기준을 만든다.

**Architecture:** 제품 코드는 수정하지 않고 `tests/math_audit/`에 결정론적 실험을 추가한다. 실험은 수학적으로 반드시 성립해야 하는 property와 현재 구현의 수치 비율을 측정하며, 결과를 `docs/audit/mathematical-audit.md`의 수식-코드 추적표에 기록한다.

**Tech Stack:** Python 3, PyTorch, pytest, NumPy, local MicroLad paper PDF/Markdown

## Global Constraints

- 공개 API, 설정 키, 체크포인트 형식의 하위 호환성은 요구하지 않는다.
- 논문보다 수학적으로 타당한 개선을 허용한다.
- 사용자가 추가한 스케일업은 핵심 기능으로 보존한다.
- 실제 모델 재학습과 품질 benchmark는 수행하지 않는다.
- 기존 작업 트리의 미커밋 변경을 덮어쓰거나 함께 커밋하지 않는다.
- 감사 단계에서는 `src/` 제품 코드를 수정하지 않는다.

---

### Task 1: 현재 동작 기준선과 수치 비교 도구

**Files:**
- Create: `tests/math_audit/__init__.py`
- Create: `tests/math_audit/helpers.py`
- Create: `tests/math_audit/test_helpers.py`

**Interfaces:**
- Consumes: scalar-valued PyTorch callables and tensors
- Produces: `central_difference(function, value, epsilon) -> Tensor`, `cosine_similarity(left, right) -> Tensor`

- [ ] **Step 1: 전체 기준선 테스트 실행**

Run: `python -m pytest -q`

Expected: 현재 pass/fail 개수와 실행 시간을 감사 보고서 작업 노트에 기록한다. 실패가 있으면 제품 코드를 바꾸지 않고 실패 목록을 기준선으로 보존한다.

- [ ] **Step 2: 수치 도구의 실패 테스트 작성**

```python
import torch

from tests.math_audit.helpers import central_difference, cosine_similarity


def test_central_difference_matches_quadratic_gradient():
    value = torch.tensor([1.5, -2.0], dtype=torch.float64)
    actual = central_difference(lambda x: x.square().sum(), value)
    assert torch.allclose(actual, 2.0 * value, atol=1e-6, rtol=1e-6)


def test_cosine_similarity_reports_parallel_and_opposite_vectors():
    vector = torch.tensor([1.0, -2.0], dtype=torch.float64)
    assert torch.allclose(cosine_similarity(vector, 3.0 * vector), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(cosine_similarity(vector, -vector), torch.tensor(-1.0, dtype=torch.float64))
```

- [ ] **Step 3: 실패 확인**

Run: `python -m pytest tests/math_audit/test_helpers.py -q`

Expected: FAIL because `tests.math_audit.helpers` does not exist.

- [ ] **Step 4: 최소 구현 작성**

```python
import torch


def central_difference(function, value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    result = torch.empty_like(value)
    flat_value = value.reshape(-1)
    flat_result = result.reshape(-1)
    for index in range(flat_value.numel()):
        delta = torch.zeros_like(flat_value)
        delta[index] = epsilon
        plus = function((flat_value + delta).reshape_as(value))
        minus = function((flat_value - delta).reshape_as(value))
        flat_result[index] = (plus - minus) / (2.0 * epsilon)
    return result


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(-1)
    right = right.reshape(-1)
    return torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
```

- [ ] **Step 5: 도구 테스트 통과 확인**

Run: `python -m pytest tests/math_audit/test_helpers.py -q`

Expected: 2 passed.

---

### Task 2: Phase, VAE, DDPM 수식 감사

**Files:**
- Create: `tests/math_audit/test_core_equations.py`

**Interfaces:**
- Consumes: `kl_divergence`, `phase_logits`, `logits_to_phase_values`, `DDPM.q_sample`, `DDPM.p_mean`
- Produces: phase 확률 의미와 DDPM 닫힌형 수식의 executable specification

- [ ] **Step 1: 수식 property 테스트 작성**

```python
import torch

from src.loss.kl import kl_divergence
from src.loss.phase import logits_to_phase_values, phase_logits
from src.models.ddpm import DDPM


class FixedNoise(torch.nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise

    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.noise.expand_as(value)


def test_kl_is_zero_for_standard_normal_and_matches_closed_form():
    mu = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float64)
    logvar = torch.tensor([[[[0.0]], [[0.0]]]], dtype=torch.float64)
    expected = torch.tensor(0.25, dtype=torch.float64)
    assert torch.allclose(kl_divergence(mu, logvar), expected)


def test_distance_logits_decode_symmetrically_between_adjacent_phases():
    value = torch.tensor([[[[0.5]]]], dtype=torch.float64)
    logits = phase_logits(value, num_phases=2, temperature=0.1)
    decoded = logits_to_phase_values(logits, num_phases=2)
    assert torch.allclose(decoded, value)


def test_q_sample_matches_closed_form():
    ddpm = DDPM(timesteps=4, beta_start=0.1, beta_end=0.2)
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-0.5]]]])
    timestep = torch.tensor([2], dtype=torch.long)
    expected = ddpm.sqrt_alphas_cumprod[2] * clean + ddpm.sqrt_one_minus_alphas_cumprod[2] * noise
    assert torch.allclose(ddpm.q_sample(clean, timestep, noise), expected)


def test_p_mean_matches_epsilon_parameterization():
    ddpm = DDPM(timesteps=4, beta_start=0.1, beta_end=0.2)
    noisy = torch.tensor([[[[0.75]]]])
    predicted_noise = torch.tensor([[[[-0.25]]]])
    timestep = torch.tensor([2], dtype=torch.long)
    expected = (noisy - ddpm.betas[2] / ddpm.sqrt_one_minus_alphas_cumprod[2] * predicted_noise) / torch.sqrt(ddpm.alphas[2])
    assert torch.allclose(ddpm.p_mean(FixedNoise(predicted_noise), noisy, timestep), expected)
```

- [ ] **Step 2: 집중 테스트 실행**

Run: `python -m pytest tests/math_audit/test_core_equations.py -q`

Expected: 4 passed. 실패 시 실제 값과 닫힌형 값의 차이를 보고서 발견 사항으로 기록하고 제품 코드는 유지한다.

- [ ] **Step 3: 기존 관련 테스트와 함께 회귀 확인**

Run: `python -m pytest tests/test_vae_loss.py tests/test_diffusion_loss.py tests/test_diffusion_models.py tests/math_audit/test_core_equations.py -q`

Expected: 모든 테스트 통과 또는 Task 1에 이미 기록된 기준선 실패만 재현.

---

### Task 3: Descriptor와 FEM 감사

**Files:**
- Create: `tests/math_audit/test_descriptors.py`

**Interfaces:**
- Consumes: `compute_volume_fraction`, `compute_tpc`, `compute_surface_area`, `DiffusivitySolver`
- Produces: analytic microstructure에 대한 descriptor 불변조건

- [ ] **Step 1: 해석 가능한 구조 테스트 작성**

```python
import torch

from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.sa import compute_surface_area
from src.predict.sds.tpc import compute_tpc
from src.predict.sds.vf import compute_volume_fraction


def test_volume_fraction_sums_to_one():
    values = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    actual = compute_volume_fraction(values, num_phases=2, temperature=0.01)
    assert torch.allclose(actual.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(actual, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-8)


def test_tpc_is_invariant_to_periodic_translation():
    values = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    shifted = torch.roll(values, shifts=(1, 2), dims=(0, 1))
    left = compute_tpc(values, num_phases=2, temperature=0.01)
    right = compute_tpc(shifted, num_phases=2, temperature=0.01)
    assert torch.allclose(left, right, atol=1e-6)


def test_surface_area_is_zero_for_homogeneous_phase_limit():
    values = torch.zeros(16, 16, dtype=torch.float64)
    actual = compute_surface_area(values, num_phases=2, temperature=0.01, kernel_size=3, sigma=1.0)
    assert torch.all(actual < 1e-8)


def test_diffusivity_normalizes_uniform_conductor_to_one():
    solver = DiffusivitySolver(4, 4, low_cond=0.001)
    actual = solver(torch.ones(4, 4))
    assert torch.allclose(actual, torch.tensor(1.0), atol=1e-6)


def test_diffusivity_is_bounded_and_monotone_for_uniform_fields():
    solver = DiffusivitySolver(4, 4, low_cond=0.01)
    low = solver(torch.zeros(4, 4))
    middle = solver(torch.full((4, 4), 0.5))
    high = solver(torch.ones(4, 4))
    assert 0.0 < low < middle < high
    assert torch.allclose(high, torch.tensor(1.0), atol=1e-6)
```

- [ ] **Step 2: 집중 테스트 실행**

Run: `python -m pytest tests/math_audit/test_descriptors.py -q`

Expected: 각 실패의 actual/expected를 수집한다. homogeneous surface area가 soft probability 때문에 작은 양수이면 temperature별 수렴값을 보고서에 기록한다.

- [ ] **Step 3: gradient 유한성 검사 추가**

```python
def test_descriptor_and_fem_gradients_are_finite():
    values = torch.tensor([[0.1, 0.9], [0.2, 0.8]], requires_grad=True)
    solver = DiffusivitySolver(2, 2, low_cond=0.01)
    loss = solver(values).sum()
    gradient, = torch.autograd.grad(loss, values)
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
```

- [ ] **Step 4: 관련 회귀 테스트 실행**

Run: `python -m pytest tests/test_predict_sds_vf.py tests/test_predict_sds_tpc.py tests/test_predict_sds_sa.py tests/test_predict_sds_diffusivity.py tests/math_audit/test_descriptors.py -q`

Expected: 모든 기존 테스트 통과; 감사 property 실패는 보고서 판정으로 전환한다.

---

### Task 4: SDS pseudo-gradient 감사

**Files:**
- Create: `tests/math_audit/test_sds_gradient.py`

**Interfaces:**
- Consumes: `sds_loss`, DDPM schedule
- Produces: 현재 surrogate gradient와 논문 Eq. 39-44 pseudo-gradient 사이의 정확한 scale 관계

- [ ] **Step 1: gradient 방향과 scale 테스트 작성**

```python
import torch

from src.models.ddpm import DDPM
from src.predict.sds.core import sds_loss
from tests.math_audit.helpers import cosine_similarity


class ConstantPrediction(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.full_like(noisy, self.value)


def test_current_sds_gradient_is_parallel_to_paper_pseudogradient():
    ddpm = DDPM(timesteps=4, beta_start=0.1, beta_end=0.2)
    latent = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    noise = torch.full_like(latent, 1.25)
    timestep = torch.tensor([2])
    loss, _ = sds_loss(latent, ConstantPrediction(0.25), ddpm, t_min=1, t_max=3, t=timestep, noise=noise)
    current, = torch.autograd.grad(loss, latent)
    alpha_bar = ddpm.alphas_cumprod[timestep].view(1, 1, 1, 1)
    paper_weight = (1.0 - alpha_bar) / alpha_bar
    paper = 2.0 * paper_weight * torch.sqrt(alpha_bar) * (0.25 - noise) / latent.numel()
    assert torch.allclose(cosine_similarity(current, paper), torch.tensor(1.0))
    expected_ratio = torch.sqrt(alpha_bar) / 2.0
    assert torch.allclose(current / paper, expected_ratio.expand_as(current))
```

- [ ] **Step 2: 집중 테스트 실행**

Run: `python -m pytest tests/math_audit/test_sds_gradient.py tests/test_predict_sds_core.py -q`

Expected: gradient 방향은 같고 current/paper 비율은 `sqrt(alpha_bar)/2`. 이 timestep-dependent scale 차이를 `의심` 또는 `의도적 개선` 후보로 기록한다.

---

### Task 5: 3D 좌표와 스케일업 불변조건 감사

**Files:**
- Create: `tests/math_audit/test_geometry_and_scale.py`

**Interfaces:**
- Consumes: slice batch helpers, `tile_grid`, `denoise_tiled_plane`, `center_start`
- Produces: 좌표 round-trip, tile coverage, 단일 tile 동등성 specification

- [ ] **Step 1: 좌표와 tile property 테스트 작성**

```python
import torch

from src.predict.scale.condition import center_start
from src.predict.scale.denoise import denoise_tiled_plane
from src.predict.scale.tiles import tile_grid
from src.predict.slices import extract_slice_batch, replace_slice_batch


class IdentityDDPM:
    posterior_variance = torch.zeros(1)

    def p_mean(self, model, value, timestep):
        return value

    def _expand(self, values, timestep, ndim):
        return values[timestep].view((timestep.shape[0],) + (1,) * (ndim - 1))


def test_slice_batch_extract_replace_round_trip_for_every_axis():
    original = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    for axis in range(3):
        indices = [0, original.shape[axis] - 1]
        selected = extract_slice_batch(original, axis, indices)
        restored = original.clone()
        replace_slice_batch(restored, axis, indices, selected + 1000)
        assert torch.equal(extract_slice_batch(restored, axis, indices), selected + 1000)


def test_tile_grid_covers_every_pixel_with_positive_weight():
    coverage = torch.zeros(9, 11, dtype=torch.int64)
    for row, col in tile_grid(9, 11, tile_size=4, overlap=2):
        coverage[row:row + 4, col:col + 4] += 1
    assert torch.all(coverage > 0)


def test_tiled_identity_denoising_matches_input_for_overlap_and_single_tile():
    value = torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6)
    timestep = torch.zeros(1, dtype=torch.long)
    for tile_size, overlap in ((6, 0), (4, 2), (3, 1)):
        actual = denoise_tiled_plane(torch.nn.Identity(), IdentityDDPM(), value, timestep, tile_size=tile_size, overlap=overlap)
        assert torch.allclose(actual, value)


def test_center_start_is_symmetric_and_integral():
    assert center_start(volume_size=8, base_size=4) == 2
    assert center_start(volume_size=7, base_size=3) == 2
```

- [ ] **Step 2: 집중 테스트 실행**

Run: `python -m pytest tests/math_audit/test_geometry_and_scale.py tests/test_predict_slices.py tests/test_predict_scale_tiles.py tests/test_predict_scale_sampler.py tests/test_predict_scale_condition.py -q`

Expected: 모든 좌표 round-trip과 coverage property 통과. 실패는 axis 또는 boundary별로 분리해 보고서에 기록한다.

---

### Task 6: 감사 보고서 작성과 리팩터링 우선순위 확정

**Files:**
- Create: `docs/audit/mathematical-audit.md`
- Modify: `docs/superpowers/plans/2026-07-10-mathematical-audit.md`

**Interfaces:**
- Consumes: Tasks 1-5의 실행 결과, 논문 Eq. 33-50, 관련 코드 위치
- Produces: 영역별 판정, 위험도, 재현 명령, 리팩터링 순서

- [ ] **Step 1: 전체 감사 테스트 실행**

Run: `python -m pytest tests/math_audit -q`

Expected: property별 pass/fail과 실제 수치를 확보한다. 실패는 숨기지 않고 보고서의 `오류` 또는 `의심` 항목에 연결한다.

- [ ] **Step 2: 보고서 작성**

보고서는 다음 확정 구조를 사용한다.

```markdown
# MicroLad 수학 감사

## 결론
## 기준선
## 위험도별 발견 사항
## 1. 데이터 표현과 phase semantics
## 2. VAE
## 3. LDM과 DDPM
## 4. 3D 재구성
## 5. SDS와 목적함수
## 6. 스케일업
## 유지할 의도적 개선
## 리팩터링 순서
## 재학습이 필요한 잔여 검증
## 재현 명령
```

각 발견 사항에는 `판정`, `위험도`, `근거 수식`, `코드 위치`, `실험`, `권고`를 모두 쓴다. 테스트가 통과한 항목도 핵심 불변조건의 근거로 기록한다.

- [ ] **Step 3: 보고서 자체 검토**

Run: `rg -n "T[B]D|T[O]DO|미[정]|확인\\s+필요" docs/audit/mathematical-audit.md`

Expected: no matches.

Run: `rg -n "^(## 1\.|## 2\.|## 3\.|## 4\.|## 5\.|## 6\.)" docs/audit/mathematical-audit.md`

Expected: 6 matches.

- [ ] **Step 4: 전체 회귀 테스트 실행**

Run: `python -m pytest -q`

Expected: Task 1 기준선보다 새로운 회귀가 없다. 감사 property 실패가 남는 경우 보고서와 정확히 일치해야 한다.

- [ ] **Step 5: 감사 산출물 커밋**

```bash
git add -f docs/audit/mathematical-audit.md docs/superpowers/plans/2026-07-10-mathematical-audit.md tests/math_audit
git commit -m "docs: audit mathematical implementation"
```

Expected: 기존 사용자 변경은 unstaged 상태로 유지되고 감사 문서와 실험만 커밋된다.
