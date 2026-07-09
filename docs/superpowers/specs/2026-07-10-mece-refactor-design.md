# MicroLad MECE 리팩터링 설계

## 목표

코드의 수학적 의미와 파일 책임을 일치시킨다. 각 package는 하나의 독립된 질문에 답하고, 같은 이유로 함께 변경되는 구현은 같은 package에 둔다. 공개 import와 설정 호환성은 유지하지 않아도 되며, 실행 entrypoint와 문서는 새 구조에 맞게 함께 바꾼다.

## 구조 원칙

- top-level package 사이에는 책임 중복이 없어야 한다.
- 같은 수학 객체의 forward 식, loss, sampler는 가까이 둔다.
- orchestration은 계산식을 소유하지 않고 조립만 한다.
- `utils.py`, `common.py`, `types.py`처럼 소유권이 불분명한 이름은 사용하지 않는다.
- 파일 이름은 구현 기법보다 도메인 역할을 말한다.
- base-size와 scale-up이 같은 수식을 사용하면 공통 package를 호출하고 복사하지 않는다.
- benchmark 없이 우열을 확정할 수 없는 수학 정책은 명시적 이름과 config 경계로 분리한다.

## 목표 package 구조

```text
src/
  api/
    options.py
    predictor.py
  phases/
    representation.py
    segmentation.py
  vae/
    model.py
    objective.py
  diffusion/
    process.py
    model.py
    objective.py
    sampler.py
  reconstruction/
    slices.py
    volume.py
    refinement.py
  guidance/
    prior.py
    objective.py
    optimization.py
    conditioning.py
    descriptors/
      volume_fraction.py
      two_point_correlation.py
      surface_area.py
    physics/
      diffusivity.py
  scaling/
    tiles.py
    blending.py
    sampling.py
    decoding.py
    refinement.py
    optimization.py
    conditioning.py
  data/
    dataset.py
    transforms.py
  training/
    vae.py
    diffusion.py
    runtime.py
    distributed.py
  io/
    images.py
  runtime/
    config.py
    distributed.py
    factories.py
    loading.py
```

`api`는 사용자가 호출하는 예측 facade만 소유한다. `runtime`은 config와 object graph 조립을 소유하고 수학 계산을 하지 않는다. `training`은 loop와 checkpoint lifecycle을 소유한다. 나머지 package는 각각 하나의 수학 또는 생성 책임을 소유한다.

## 수학 정책

### Phase 표현

VAE 학습은 categorical logits와 cross entropy를 유지한다. 현재 inference가 사용하는 scalar relaxation은 숨기지 않고 `logits_to_relaxed_labels`로 명명한다. categorical probability를 보존하는 API와 최종 `argmax` label API를 함께 제공한다.

전체 3D optimizer를 simplex logits로 바꾸는 일은 checkpoint 기반 품질 benchmark가 필요하므로 이번 구조 리팩터링에서 기본 동작으로 강제하지 않는다. 대신 descriptor가 probability를 직접 받을 수 있는 경계를 만들고, scalar relaxation 사용 위치를 `phases` package로 한정한다.

### KL과 latent 통계

현재 element-mean KL과 deterministic mean latent를 기준선으로 유지하되 이름과 문서에 reduction을 명시한다. latent normalization은 runtime config에 암묵적으로 넣지 않고 후속 benchmark 항목으로 남긴다.

### SDS weighting

현재 `sigma_squared` weighting을 이름으로 드러낸다. 논문 weighting과 방향은 같지만 timestep scale이 다르므로 조용히 바꾸지 않는다. weighting 계산을 `guidance.prior` 한 곳에 둬 후속 비교가 가능하게 한다.

### Scale overlap

output stitching과 local loss reduction이 같은 normalized overlap 의미를 사용하도록 한다. anchor loss는 stitched decoded slice에서 한 번 계산한다. local descriptor는 명시적 local scope에서만 patch 평균을 사용한다. SDS tile loss는 latent coverage weight를 받아 overlap 횟수 때문에 중앙 gradient가 커지지 않게 한다.

## 주요 interface

- `PatchVAE.decode_logits(latent) -> Tensor[B, P, H, W]`
- `PatchVAE.decode_probabilities(latent) -> Tensor[B, P, H, W]`
- `PatchVAE.decode_relaxed_labels(latent) -> Tensor[B, 1, H, W]`
- `DDPMProcess.q_sample(clean, timestep, noise) -> noisy`
- `DDPMProcess.p_mean(model, noisy, timestep) -> mean`
- `score_distillation_loss(..., weighting=SDSWeighting.SIGMA_SQUARED, spatial_weight=None)`
- `DescriptorScope.LOCAL_PATCH | DescriptorScope.WHOLE_SLICE`
- `normalized_tile_weights(height, width, tile_size, overlap) -> per-tile weight maps`
- `load_predictor(run_dir, device) -> Predictor`

## Error 처리

shape, dtype, finite-value 검사는 그 값을 처음 소유하는 package 경계에서 수행한다. orchestration layer가 같은 검사를 반복하지 않는다. 내부에서 발생한 모든 예외를 포괄적인 `ValueError`로 다시 감싸지 않고, checkpoint/config처럼 외부 경계에 추가 문맥이 필요한 경우에만 원인을 연결해 다시 발생시킨다.

## 테스트 전략

1. 기존 390개 테스트와 72개 subtest를 characterization 기준선으로 사용한다.
2. 이동 전 import를 새 package import로 바꾼 테스트가 먼저 실패하는 것을 확인한다.
3. 수학 함수 이동은 감사 테스트 20개로 동등성을 확인한다.
4. overlap-normalized loss는 중앙과 corner gradient가 같아지는 실패 테스트를 먼저 작성한다.
5. 각 package 이동 후 관련 집중 테스트를 실행하고, 마지막에 전체 suite를 실행한다.
6. entrypoint import, README 예제, config round-trip을 별도로 확인한다.

## 실행 순서

1. phase, VAE, diffusion core를 이동해 수학 기초 package를 확정한다.
2. descriptor, physics, SDS를 guidance로 이동하고 이름을 명확히 한다.
3. reconstruction과 scaling을 분리하고 overlap gradient를 수정한다.
4. training과 runtime 조립을 분리한다.
5. Predictor facade를 줄이고 새 package를 조립한다.
6. tests, scripts, notebook, README import를 새 구조에 맞춘다.
7. 전체 테스트, compile, import smoke check, diff 검토를 수행한다.

## 완료 기준

- `models`, `loss`, `predict`처럼 서로 다른 책임을 섞는 기존 package가 제거된다.
- 모든 production Python 파일은 설명 가능한 단일 책임을 가진다.
- base와 scale guidance가 공통 objective와 weighting implementation을 사용한다.
- overlap 설정이 pixel별 local loss strength를 바꾸지 않는다.
- phase categorical/relaxed/final label 변환이 이름과 한 package에 한정된다.
- maintained training entrypoint와 predictor loading 경로가 새 구조에서 동작한다.
- 전체 테스트와 감사 테스트가 통과한다.
