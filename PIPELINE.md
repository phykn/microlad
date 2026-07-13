# 통합 3D 생성 파이프라인

## 목표

하나 이상의 지정 단면과 목표 phase fraction을 만족하면서, XY·XZ·YZ 어느 축으로
잘라도 학습 데이터와 같은 형태를 보이는 연속적인 3D categorical volume을 만든다.
조건 단면은 그대로 복사하지 않고 soft loss로 맞춘다.

사용자에게 보이는 생성 경로는 하나다. L-MPDD, SDS, latent critic, 앵커,
phase fraction, Refine을 서로 대체하는 모드로 제공하지 않는다. 각 기능은 같은
L-MPDD 후보를 개선하거나 평가하는 단계다.

## 현재 문제

기존 conditional SliceGAN은 L-MPDD를 실행하지 않고 random noise에서 별도의 3D
latent를 생성했다. Diffusion은 2D reference 생성에만 사용됐으며, SliceGAN을
선택하면 Joint·SDS·Refine도 실행되지 않았다. 따라서 좋은 SliceGAN 결과와
L-MPDD가 만든 3D 초기 구조를 하나의 결과에서 활용할 수 없었다.

기존 SliceGAN 앵커 loss도 `output_index // downsample_factor`의 latent plane 하나를
독립적으로 decode했다. 최종 decoder는 latent plane을 보간하고 세 축 probability를
합의하므로, 조건 loss와 최종 단면이 서로 다른 좌표계와 함수를 사용했다.

## 데이터 흐름

```text
reference / anchor images
    -> image-space augmentation
    -> frozen categorical VAE mean latent
    -> real latent bank

frozen 2D diffusion
    -> L-MPDD latent candidates
    -> axis slice/crop
    -> fake latent bank
    -> online critic warm-up and validation
    -> frozen critic

selected L-MPDD latent Z0
    -> Z = Z0 + alpha * R(Z0)
    -> joint latent optimization
       - latent SDS
       - frozen critic guidance
       - exact decoded anchor loss
       - decoded phase-fraction loss
       - paper descriptors (TPC, surface area, diffusivity)
       - decoded continuity and latent preservation
    -> shared tri-axis categorical probability decoder
    -> Refine candidates (0, 1, 2 sweeps)
    -> budget-limited categorical calibration
    -> strict hard-volume quality gate
    -> best feasible volume
```

VAE와 diffusion weight는 prediction 중 갱신하지 않는다. 온라인으로 학습하는 것은
critic과 zero-initialized 3D residual refiner뿐이다.

현재 reference augmentation과 최종 morphology 평가는 세 축이 같은 통계를 따른다는
등방성 가정을 사용한다. 축마다 다른 조직을 요구하는 anisotropic mode는 지원하지
않으며, 그 경우에는 축별 reference bank와 축별 target을 별도 설계해야 한다.

## 책임

### Reconstruction

- L-MPDD sampling과 tri-axis categorical decoding을 소유한다.
- sampling은 latent를 반환하고 decoding은 명시적인 다음 단계로 둔다.
- exact anchor loss와 최종 출력이 같은 decoder를 사용한다.

### Critic guidance

- reference latent bank와 L-MPDD fake bank를 준비한다.
- latent critic을 warm-up하고 held-out real/fake margin을 검증한다.
- critic을 고정한 뒤 joint optimizer에 read-only guidance로 제공한다.
- random-noise 3D generator는 소유하지 않는다.

### Joint guidance

- `Z0`를 보존하는 zero-initialized residual refiner를 소유한다.
- SDS, critic, anchor, fraction, 원 논문 descriptor, continuity, preservation
  loss를 한 optimizer에서 합산한다.
- 원본 `Z0`와 중간 checkpoint를 후보로 보존한다.

### Finalization

- Refine sweep 수가 다른 probability 후보를 만든다.
- calibration 전후 fraction, 변경 voxel 비율, 앵커 변화, morphology 변화를 기록한다.
- hard gate를 통과한 후보만 선택한다. 합격 후보가 없고 strict mode라면 실패한다.

## Critic 계약

Real은 reference 이미지를 변환한 뒤 VAE posterior mean으로 encode한 latent다. VAE의
회전 equivariance를 가정하지 않으므로 latent를 먼저 회전하지 않는다. Clean 2D
diffusion latent를 추가할 경우 real-like reference로만 사용한다.

Fake는 여러 L-MPDD latent volume의 세 축 slice/crop이다. 한 `Z0`만 사용해 sample
고유 특징을 외우지 않도록 학습용과 검증용 crop을 분리한다. Critic은 다음 조건을
충족한 경우에만 residual guidance에 사용한다.

- held-out real score가 held-out fake score보다 높다.
- score와 입력 gradient가 finite다.
- phase fraction만 바꾼 샘플보다 반복 slab나 checkerboard 손상에 더 민감하다.

첫 구현은 critic warm-up 후 완전히 고정한다. Residual과 critic을 매 step 동시에
갱신하지 않는다.

## Exact anchor와 공간 척도

앵커는 최종 출력 index와 footprint로 정의한다. 앵커 loss는 full decoder와 동일한
latent-plane interpolation과 geometric probability consensus를 거친 probability
slice에 적용한다. Target label을 voxel에 덮어쓰지 않는다.

Continuity loss와 transition 진단은 latent plane 간격이 아니라 decoded output
voxel의 실제 인접 slice에서 계산한다. VAE downsample factor가 4일 때 인접 latent
plane을 1-voxel 이웃 통계와 직접 비교하지 않는다.

## Calibration과 후보 선택

Calibration은 마지막에 후보당 한 번만 수행한다. 앵커 target이 아니라 calibration
직전 모델이 선택한 앵커 영역 label을 보호한다. 다음 값을 항상 분리해 기록한다.

- pre/post calibration phase fraction
- changed voxel fraction
- anchor mismatch delta
- transition, run-profile, boundary-jump delta

변경 voxel 비율이 budget을 넘으면 해당 후보는 실패다. 후보 선택 우선순위는 다음과
같다.

1. 최대 anchor mismatch
2. phase-fraction tolerance와 calibration budget
3. 반복 단면, boundary cutoff, 축 collapse
4. transition과 run-profile
5. 합격 후보 사이의 morphology score

하나의 낮은 scalar score가 앵커나 fraction 실패를 상쇄하지 못한다.

## Scale-up 경계

Scale-up도 L-MPDD latent에서 시작하고 공통 categorical decoder와 최종 evaluator를
사용한다. 다만 large tiled decoder는 현재 gradient를 보존하지 않으므로, 첫 통합
버전의 exact latent Joint는 VAE 기본 크기에서 완성한다. 큰 volume은 기존 tiled
SDS/anchor optimizer를 같은 상위 파이프라인의 scale 단계로 유지한다.

Large latent Joint는 다음 두 조건을 충족한 뒤 확장한다.

- tiled tri-axis decoder와 exact anchor region decode가 gradient를 보존한다.
- base와 large decoder가 같은 좌표와 probability 의미를 갖는 회귀 테스트를 통과한다.

## 제거한 레거시

다음 구현은 통합 과정에서 삭제했고 호환 계층도 두지 않았다.

- random-noise `SliceGANGenerator`
- generator/noise 후보 선택과 조건 미세조정
- SliceGAN 전용 tiled generator renderer
- SliceGAN을 L-MPDD·Joint·SDS와 상호 배타적으로 만드는 Predictor 분기
- `SliceGAN*Config`, `config/slicegan.yaml`, `load_slicegan_config`
- SliceGAN 전용 통계 이름과 노트북 저장 계약

WGAN-GP objective, latent slice sampling, 2D latent critic은 critic guidance로
이동했다. 독립 SDS 모드에서 실제로 쓰던 diffusion prior는 공통 guidance로,
scale slice schedule은 scaling으로 옮기고 나머지 slice optimizer도 삭제했다.

## 검증

예측의 L-MPDD sampling, latent critic warm-up, Joint guidance, scale-up
sampling/guidance는 `progress` 설정을 공유하고 노트북에서는 기본적으로 진행률을
일반 텍스트 `tqdm`으로 표시한다. critic은 loss와 margin을, Joint는 전체 loss와
활성화된 anchor·critic·fraction loss를 일정 간격으로 갱신한다.

구조 검증:

- Predictor의 모든 결과가 L-MPDD sampling을 거친다.
- 독립 SliceGAN generator와 전용 public option이 남아 있지 않다.
- exact anchor loss와 final decode의 동일 좌표 결과를 비교한다.
- Joint의 tri-axis VAE decode는 `decode_batch_size`가 숫자이면 plane batch와
  gradient checkpointing으로 peak activation memory를 제한한다. 큰 GPU에서는
  `decode_batch_size: null`로 지정해 checkpointing 없이 축별 plane 전체를 한
  batch로 처리한다. 두 모드는 동일한 decoder 의미를 사용한다.

수학 검증:

- residual output layer의 초기 출력은 정확히 0이다.
- critic warm-up 중 VAE·diffusion·`Z0`에는 gradient가 없다.
- frozen critic guidance는 residual parameter에 finite gradient를 전달한다.
- calibration budget과 strict gate가 실패 후보 반환을 막는다.

회귀 검증:

- categorical VAE, L-MPDD, scale-up, 다중 축 앵커, phase fraction API를 보존한다.
- `.venv\\Scripts\\python.exe -m pytest -q` 전체 테스트를 통과한다.
- `03_predict.ipynb`는 통합 기본 크기 예제로, `04_scale_up.ipynb`는 같은 상위
  파이프라인의 scale 예제로 실행 가능해야 한다.
