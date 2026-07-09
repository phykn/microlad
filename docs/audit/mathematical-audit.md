# MicroLad 수학 감사

## 결론

현재 구현의 DDPM 전방·역방향 식, 기본 descriptor, FEM 정규화, 세 축 좌표 변환, 타일 coverage와 weighted denoising은 작은 합성 입력에서 일관되게 동작한다. 사용자 개선인 categorical VAE reconstruction과 타일 mean 결합 후 단일 noise 주입도 유지할 가치가 있다.

리팩터링에서 먼저 고쳐야 할 수학 위험은 두 가지다.

1. categorical logits를 scalar phase 기대값으로 축약한 뒤 거리 기반 확률로 다시 만드는 과정이 phase 정보를 잃는다.
2. 겹치는 스케일업 tile의 loss를 단순 평균해 중앙 pixel이 경계 pixel보다 더 큰 gradient를 받는다.

그다음으로 SDS timestep weighting, KL reduction, latent 통계 계약을 명시해야 한다. 이 항목들은 즉시 틀렸다고 단정할 수는 없지만 현재 hyperparameter 의미가 수식과 파일 경계에 드러나지 않는다.

## 리팩터링 반영 결과

- phase categorical probability, relaxed label, final label 변환을 `src/phases` 한 package로 모으고 정보 손실이 있는 변환을 `logits_to_relaxed_labels`로 명명했다.
- scale output과 SDS local prior에 normalized tile ownership을 적용했다.
- anchor objective는 겹치는 tile마다 반복하지 않고 stitched slice에서 한 번 계산한다.
- 감사에서 4배였던 중앙/corner anchor gradient가 같은 값이 되는 회귀 테스트를 추가했다.
- `models/loss/predict` 기술 계층을 제거하고 수학·생성 책임별 package로 이동했다.
- KL reduction, latent normalization, SDS timestep weighting은 실제 checkpoint benchmark가 필요해 기준선 정책을 유지하고 한 파일에 격리했다.

## 기준선

- 작업 공간: `codex/math-audit-refactor` worktree
- 사용자 미커밋 변경 7개 파일을 원본과 같은 diff로 복제했다.
- RGB phase 이미지는 첫 번째 채널을 label plane으로 사용한다. 이 사용자 의도에 맞게 낡은 테스트를 수정했다.
- 전체 기준선: `370 passed, 72 subtests passed`
- 추가 감사 실험: `20 passed`
- 실제 모델 재학습과 생성 품질 benchmark는 수행하지 않았다.

재현 명령:

```powershell
python -m pytest tests/math_audit -q
python -m pytest -q
```

## 위험도별 발견 사항

| 위험도 | 판정 | 발견 사항 | 우선 조치 |
| --- | --- | --- | --- |
| 높음 | 오류 | categorical 분포를 scalar phase 기대값으로 축약하면 다른 phase로 해석될 수 있다 | phase probability를 channel 축으로 유지한다 |
| 높음 | 오류 | 겹치는 tile의 local loss gradient가 coverage 횟수에 비례한다 | loss에도 정규화된 공간 weight를 적용한다 |
| 중간 | 의심 | 현재 SDS gradient는 논문 pseudo-gradient와 평행하지만 timestep별 scale이 다르다 | weighting 정책을 이름과 수식으로 고정한다 |
| 중간 | 의심 | KL이 latent element 전체 평균이라 latent 크기가 변해도 값이 변하지 않는다 | reduction과 beta 의미를 명시한다 |
| 중간 | 구조 위험 | base SDS와 scale SDS의 objective 조립·검증·통계 코드가 중복된다 | 공통 guidance pipeline으로 합친다 |
| 중간 | 구조 위험 | `predictor.py`, `sds/optimize.py`, `scale/sds.py`가 orchestration과 수학을 함께 가진다 | 책임별 package로 분리한다 |
| 낮음 | 의도적 개선 | FEM에서 0 conductivity를 0.001로 올려 선형계 특이성을 피한다 | numerical floor를 공개된 solver 설정으로 명명한다 |

## 1. 데이터 표현과 phase semantics

### categorical 확률의 scalar 축약

- 판정: `오류`
- 위험도: `높음`
- 코드: `src/phases/representation.py:82`, `src/vae/model.py:199`, `src/predict/sds/phase.py`
- 현재 흐름: decoder logits → softmax 기대 phase 번호 → 거리 기반 soft phase probability
- 문제: phase는 순서형 수치가 아니라 nominal category다. 기대 phase 번호는 categorical mixture를 보존하지 않는다.
- 실험: logits가 phase 0과 2에 각각 0.5, phase 1에 거의 0을 주도록 만들면 scalar 기대값은 1이 된다. 이를 거리 기반 확률로 복원하면 phase 1 확률이 0.99보다 커졌다.
- 근거 테스트: `test_scalar_phase_expectation_can_turn_bimodal_uncertainty_into_other_phase`
- 권고: 학습, decode, descriptor, anchor guidance 사이에서 `[B, P, H, W]` logits 또는 probability를 유지한다. 최종 파일 출력에서만 `argmax` phase label로 바꾼다. 연속 scalar volume이 필요한 optimizer에는 simplex logits를 직접 최적화한다.

### RGB phase 입력

- 판정: `의도적 개선`
- 위험도: `낮음`
- 코드: `src/io/image.py:19`
- 현재 흐름: 3차원 이미지 배열은 첫 번째 채널을 phase label plane으로 사용한다.
- 검증: 서로 다른 세 channel을 가진 RGB PNG에서 첫 channel과 결과가 정확히 같았다.
- 권고: 함수 이름과 docstring에 `first channel` 계약을 드러내고 grayscale intensity loader와 분리한다.

## 2. VAE

### categorical reconstruction

- 판정: `의도적 개선`
- 위험도: `낮음`
- 코드: `src/vae/model.py:190`, `src/vae/objective.py:8`, `src/phases/representation.py:60`
- 현재 흐름: decoder가 phase별 logits를 내고 정수 target에 cross entropy를 적용한다.
- 평가: phase label을 연속 intensity MSE로 다루는 것보다 nominal category 의미에 맞다.
- 권고: 이 개선을 유지하고 inference도 categorical channel 표현과 일치시킨다.

### KL reduction

- 판정: `의심`
- 위험도: `중간`
- 코드: `src/vae/objective.py:4`
- 현재 흐름: batch, channel, 공간 축을 모두 한 번에 평균한다.
- 실험: 같은 비정상 posterior element를 `1 x 1 x 1`에서 `4 x 8 x 8`로 반복해도 KL 값이 같았다.
- 해석: reconstruction CE도 pixel 평균이므로 현재 식 자체는 최적화 가능한 정규화다. 다만 표준 ELBO의 sample별 latent 합과 다르며 `beta=1`의 의미가 latent와 image 해상도 비율에 의존한다.
- 권고: `mean_per_latent_element` 또는 `sum_per_sample` 중 하나를 선택해 함수 이름, config, 문서에 명시하고 짧은 beta sensitivity 실험을 수행한다.

### diffusion latent 계약

- 판정: `의도적 개선`
- 위험도: `중간`
- 코드: `src/training/diffusion.py:81`
- 현재 흐름: VAE posterior sample이 아니라 encoder mean을 LDM 학습 데이터로 사용한다.
- 평가: 결정론적 latent dataset이라는 장점이 있으나 latent scale 통계가 저장되지 않는다.
- 권고: mean latent 사용을 유지하되 학습 run에 channel별 mean/std 또는 단일 scale을 기록하고 train/inference에서 동일하게 적용할지 benchmark로 결정한다.

## 3. LDM과 DDPM

### forward와 reverse 식

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/diffusion/process.py:69`, `src/diffusion/process.py:88`, `src/diffusion/objective.py:8`
- 검증: `q_sample`은 `sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise`와 같았다. `p_mean`은 epsilon parameterization의 DDPM reverse mean과 같았다.
- 경계: timestep 0의 posterior variance는 0이며 noise가 추가되지 않는다.

### tiled denoising

- 판정: `의도적 개선`
- 위험도: `낮음`
- 코드: `src/scaling/denoising.py:9`
- 현재 흐름: 각 tile의 reverse mean을 weighted blend한 뒤 plane 전체에 noise를 한 번 추가한다.
- 검증: identity reverse mean에서 단일 tile, overlap tile 모두 원본과 같았다.
- 평가: tile별 독립 noise를 blend하는 것보다 하나의 global transition variance를 유지한다.

## 4. 3D 재구성

### 세 축 slice 좌표

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/reconstruction/slices.py`, `src/reconstruction/volume.py:37`, `src/reconstruction/refinement.py:7`
- 검증: 세 axis 모두 batch extract 후 replace round-trip이 정확했다. 큰 volume의 center offset도 짝수 차이에서 대칭이었다.
- 평가: axis permutation과 inverse permutation이 현재 tensor convention에서 일치한다.

### 세 축 평균

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/reconstruction/volume.py:37`, `src/reconstruction/refinement.py:34`
- 현재 흐름: 세 orthogonal decode 결과에 동일한 `1/3` weight를 준다.
- 평가: 논문 Eq. 35와 일치한다. anisotropic model을 도입한다면 axis별 weight가 별도 계약이 되어야 한다.

## 5. SDS와 목적함수

### SDS pseudo-gradient weighting

- 판정: `의심`
- 위험도: `중간`
- 코드: `src/guidance/prior.py:7`
- 논문 기준: Eq. 39-44의 `kappa(t)=(1-alpha_bar)/alpha_bar`와 frozen noise predictor pseudo-gradient
- 현재 구현: `sigma^2 * latent * (predicted_noise-noise)` surrogate
- 실험: 현재 gradient와 논문 식에서 유도한 pseudo-gradient의 cosine similarity는 1이었다. 크기 비율은 `sqrt(alpha_bar)/2`였다.
- 해석: 상수 `1/2`는 learning rate로 흡수할 수 있지만 `sqrt(alpha_bar)`는 timestep마다 달라 noise level weighting을 바꾼다.
- 권고: `paper_weight`, `dreamfusion_weight`, `sigma_squared_weight`처럼 정책을 명시하고 기본값을 수식과 연결한다. 변경 전 짧은 품질 benchmark가 필요하다.

### volume fraction

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/guidance/descriptors/volume_fraction.py:54`
- 검증: soft phase probability의 phase 합과 평균 volume fraction 합이 1이고 반반 구조가 `[0.5, 0.5]`였다.
- 조건: scalar phase relaxation을 channel probability로 바꾸면 이 함수는 probability 평균으로 단순화해야 한다.

### two-point correlation

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/guidance/descriptors/two_point_correlation.py:62`
- 검증: FFT autocorrelation과 radial bin 결과가 periodic translation에 불변이었다.
- 한계: FFT는 periodic boundary를 가정하고 corner 거리까지 radial bin을 만든다. target과 prediction이 같은 함수라 내부 일관성은 있지만 외부 S2 구현과 비교할 때 cutoff를 명시해야 한다.

### relative surface area

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/guidance/descriptors/surface_area.py:69`
- 검증: homogeneous phase probability에서 Gaussian smoothing 후 total variation이 0이었다.
- 한계: pixel spacing이 1로 고정되어 있어 물리 단위가 필요한 데이터에는 voxel size 인자가 필요하다.

### differentiable diffusivity

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/guidance/physics/diffusivity.py:16`, `src/guidance/physics/diffusivity.py:220`
- 검증: uniform high conductor는 1로 정규화됐고 uniform conductivity 증가에 단조적이었다. 작은 비균질장에서 gradient는 유한하고 0이 아니었다.
- 의도적 차이: `low_cond=0`을 0.001로 바꿔 singular stiffness matrix를 피한다.
- 한계: dense stiffness matrix와 dense solve는 큰 grid에서 비용이 급격히 증가한다. 현재 downsample solver 크기를 명시적으로 관리해야 한다.

## 6. 스케일업

### tile coverage와 output blend

- 판정: `정상`
- 위험도: `낮음`
- 코드: `src/scaling/tiles.py`, `src/scaling/blending.py`, `src/scaling/denoising.py`
- 검증: 비정수 stride 끝에서도 모든 pixel이 한 번 이상 덮였다. normalized Hann blend는 identity field를 보존했다.

### overlap loss의 공간 편향

- 판정: `오류`
- 위험도: `높음`
- 코드: `src/scaling/local_objective.py:415`, `src/scaling/local_objective.py:541`, `src/scaling/local_objective.py:546`, `src/scaling/local_objective.py:704`
- 현재 흐름: decoded output은 Hann window와 weight sum으로 정규화하지만 SDS, anchor, descriptor loss는 tile별 scalar를 더한 뒤 tile count로만 나눈다.
- 실험: `4 x 4` image를 `3 x 3`, overlap 2 tile로 나눠 동일한 anchor loss를 적용했을 때 네 tile에 포함되는 중앙 pixel의 gradient가 한 tile에만 포함되는 corner보다 정확히 4배 컸다.
- 영향: volume 크기, 마지막 tile 배치, overlap 설정에 따라 같은 pixel objective의 유효 weight가 달라진다. seam 감소용 overlap이 optimization strength까지 바꾼다.
- 권고: 각 tile loss를 normalized blend window 또는 inverse coverage map으로 pixelwise reduction한다. SDS latent loss에는 latent-resolution coverage map을 사용한다. global descriptor는 stitched image에서 한 번 계산하고, local descriptor 모드만 명시적으로 patch 평균을 사용한다.

### local과 global descriptor 의미

- 판정: `의도적 개선`
- 위험도: `중간`
- 코드: `src/api/predictor.py:523`, `src/scaling/local_objective.py:232`
- 현재 흐름: target image가 VAE base size이면 각 tile descriptor 분포를 맞추고, target이 큰 volume size이면 stitched 전체 slice descriptor를 맞춘다.
- 평가: 두 사용 사례 모두 타당하지만 `descriptor_tile_size is None`이라는 간접 표현 때문에 의미가 숨겨져 있다.
- 권고: `DescriptorScope.LOCAL_PATCH`와 `DescriptorScope.WHOLE_SLICE`처럼 명시적 정책으로 바꾼다.

## 유지할 의도적 개선

- phase reconstruction의 categorical logits와 cross entropy
- diffusion 학습에서 결정론적 VAE mean latent 사용
- 세 축 결과의 동시 voxel 평균
- tiled reverse mean을 먼저 blend하고 plane noise를 한 번만 추가하는 방식
- Hann weighted output stitching
- FEM의 positive conductivity floor
- base-size target을 local patch distribution으로 사용하는 스케일업 모드
- RGB phase 이미지의 첫 번째 채널 사용 계약

## MECE 리팩터링 순서

각 package는 하나의 질문에만 답하고, 함께 변경되는 유사 개념을 같은 package에 둔다.

1. `phases`: phase label, logits, probability, 최종 quantization을 소유한다. 다른 package가 scalar phase 변환을 직접 만들지 못하게 한다.
2. `autoencoder`: VAE architecture와 ELBO만 소유한다. phase 변환은 `phases` interface를 사용한다.
3. `diffusion`: noise schedule, training objective, reverse transition, 2D sampler를 소유한다.
4. `reconstruction`: slice geometry, 세 축 L-MPDD, decode와 refinement를 소유한다.
5. `guidance`: SDS prior와 descriptor/physics objective 조립을 소유한다. `descriptors`와 `physics`는 그 하위의 서로 배타적인 계산 그룹으로 둔다.
6. `scaling`: tile geometry, overlap normalization, 큰 volume reconstruction adapter를 소유한다. guidance 수식을 복사하지 않고 `guidance` interface를 호출한다.
7. `conditioning`: anchor와 target 준비 및 좌표 정렬을 소유한다.
8. `training`, `data`, `io`: 실행 loop, dataset 변환, 외부 파일 경계를 각각 소유한다.
9. 최상위 `api`: config를 typed object로 만들고 위 package를 조립한다. 현재 `build.py`와 `predictor.py`의 orchestration을 이 경계로 이동한다.

파일 이름은 역할을 그대로 표현하고 `utils.py`, `common.py`, `types.py` 같은 포괄 이름은 제거한다. 공통 코드라는 이유만으로 묶지 않고, 그 코드의 주 책임 package에 둔다.

## 재학습이 필요한 잔여 검증

- KL reduction과 beta 조합이 reconstruction 및 latent prior에 미치는 영향
- latent channel mean/std 정규화의 LDM 품질 영향
- SDS timestep weighting 세 정책의 목표 도달 속도와 realism trade-off
- scalar phase optimizer를 simplex logits optimizer로 바꾼 뒤 descriptor control 품질
- overlap-normalized scale guidance가 seam과 목표 수렴에 미치는 영향
- local patch descriptor와 whole-slice descriptor의 scale-up 일반화 차이

이 항목은 작은 합성 실험만으로 우열을 확정할 수 없다. 구조 리팩터링에서는 정책을 분리하고 현재 동작을 선택 가능한 기준선으로 남긴 뒤, 실제 checkpoint로 짧은 benchmark를 수행해야 한다.

## 재현 명령

```powershell
python -m pytest tests/math_audit/test_helpers.py -q
python -m pytest tests/math_audit/test_core_equations.py -q
python -m pytest tests/math_audit/test_descriptors.py -q
python -m pytest tests/math_audit/test_sds_gradient.py -q
python -m pytest tests/math_audit/test_geometry_and_scale.py -q
python -m pytest tests/math_audit -q
python -m pytest -q
```
