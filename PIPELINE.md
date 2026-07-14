# 통합 3D 생성 파이프라인

## 목표

하나 이상의 지정 단면과 목표 phase fraction을 만족하면서, XY·XZ·YZ 어느 축으로
잘라도 학습 데이터와 같은 형태를 보이는 연속적인 3D categorical volume을 만든다.
조건 단면은 그대로 복사하지 않고 soft loss로 맞춘다.

사용자에게 보이는 생성 경로는 하나다. L-MPDD, SDS, 선택적 latent critic, 앵커,
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
categorical 2D training images
    -> image-space augmentation
    -> frozen categorical VAE mean latent
    -> WGAN real latent patches

2D conditional WGAN generator
    -> fake latent patches
    -> phase-conditioned WGAN-GP critic training
    -> generator and critic checkpoint

frozen 2D diffusion
    -> L-MPDD latent Z0

selected L-MPDD latent Z0
    -> Z = Z0 + alpha * R(Z0)
    -> joint latent optimization
       - latent SDS
       - frozen critic guidance
       - exact decoded anchor loss
       - decoded global phase-fraction loss
       - optional per-slice fraction descriptor
       - paper descriptors (TPC, surface area, diffusivity)
       - decoded continuity and latent preservation
       - tri-axis decoded phase agreement
    -> shared tri-axis categorical probability decoder
    -> Refine candidates (0, 1, 2 sweeps)
    -> budget-limited categorical calibration
    -> hard-volume quality evaluation
    -> best feasible or least-violation volume
```

VAE, diffusion, WGAN critic weight는 prediction 중 갱신하지 않는다. 예측마다
최적화하는 것은 zero-initialized 3D residual refiner뿐이다. WGAN generator는 critic
사전학습과 생성 결과 평가에만 사용하며 prediction에는 참여하지 않는다.

현재 reference augmentation과 최종 morphology 평가는 세 축이 같은 통계를 따른다는
등방성 가정을 사용한다. 축마다 다른 조직을 요구하는 anisotropic mode는 지원하지
않으며, 그 경우에는 축별 reference bank와 축별 target을 별도 설계해야 한다.

## 책임

### Reconstruction

- L-MPDD sampling과 tri-axis categorical decoding을 소유한다.
- sampling은 latent를 반환하고 decoding은 명시적인 다음 단계로 둔다.
- exact anchor loss와 최종 출력이 같은 decoder를 사용한다.

### Critic guidance

- 기존 categorical patch loader와 고정 VAE로 real latent를 만든다.
- 2D conditional generator와 critic을 SliceGAN 계열 WGAN-GP objective로 사전학습한다.
- generator와 critic 모두 phase fraction을 조건으로 받는다.
- GAN run은 VAE checkpoint만 이어받고 generator·critic을 함께 저장한다.
- prediction은 `predict.yaml`에서 VAE·diffusion·GAN run을 각각 명시해 조합한다.
- prediction은 generator를 사용하지 않고 critic만 read-only guidance로 사용한다.
- critic은 base Joint와 scale guidance의 세 축 latent crop에 동일하게 적용한다.

### Joint guidance

- `Z0`를 보존하는 zero-initialized residual refiner를 소유한다.
- SDS, critic, anchor, global fraction, 선택적인 slice descriptor, continuity,
  preservation loss를 한 optimizer에서 합산한다.
- 원본 `Z0`와 중간 checkpoint를 후보로 보존한다.

### Finalization

- Refine sweep 수가 다른 probability 후보를 만든다.
- calibration 전후 fraction, 변경 voxel 비율, 앵커 변화, morphology 변화를 기록한다.
- 통과 후보가 있으면 그중 최선을, 없으면 위반량이 가장 작은 결과를 반환한다.

## Critic 계약

Real은 categorical 2D patch를 고정 VAE posterior mean으로 encode한 latent다. Fake는
phase fraction과 noise를 입력받은 2D generator의 latent다. Critic은 real과 fake에
같은 fraction condition을 받아 조성 자체가 아니라 같은 조성에서의 형태 차이를
평가한다. 입력은 channel별 공간 평균을 제거하고 slice 전체 RMS로 정규화한다.

학습은 critic을 `critic_steps`번 갱신한 뒤 generator를 한 번 갱신하는 WGAN-GP다.
Generator는 adversarial loss와 VAE categorical decode의 fraction loss를 함께 받는다.
Prediction에서는 두 네트워크를 동시에 갱신하지 않으며 critic을 완전히 고정한다.

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

변경 voxel 비율이 budget을 넘으면 해당 후보를 위반 후보로 기록한다. 통과 후보가
없어도 생성물을 버리지 않고 아래 gate 순서의 사전식 순위가 가장 좋은 후보를
반환한다. 후보 선택 우선순위는 다음과 같다.

1. 최대 anchor mismatch
2. phase-fraction tolerance와 calibration budget
3. 반복 단면, boundary cutoff, 축 collapse
4. transition과 run-profile
5. 합격 후보 사이의 morphology score
6. 나머지 품질이 같으면 `latent delta RMS / base std`가 작은 후보

하나의 낮은 scalar score가 앵커나 fraction 실패를 상쇄하지 못한다.

## Scale-up 경계

Scale-up도 L-MPDD latent에서 시작하고 공통 categorical decoder와 최종 evaluator를
사용한다. 큰 volume의 guidance는 단일 3D latent residual을 최적화하고, SDS와 2D
descriptor만 학습 크기의 latent crop으로 계산한다. 따라서 과거처럼 float phase ID
단면을 수정한 뒤 한 축씩 volume에 덮어쓰지 않는다. anchor는 large L-MPDD 주입과
최종 tiled tri-axis consensus의 categorical NLL로 사용하며 target label을 직접
복사하지 않는다. 매 step에는 앵커 하나의 부분 영역을 순환해 미분하므로 전체 exact
loss와 같은 목적을 유지하면서 GPU 메모리를 제한한다.

초기 large L-MPDD latent는 후보 0이고 `scale.checkpoint_every`마다 중간 latent를
후보로 보존한다. 각 후보는 동일한 tiled decoder와 tri-axis geometric consensus로
soft probability가 된 뒤, base와 같은 strength/anchor-strength Refine 및 anchor 보호
calibration을 거쳐 비교된다. anchor와 fraction gate를 통과한 후보 사이에서는 반복,
boundary, run-profile, transition 순의 morphology가 먼저 선택된다.

`scale.decode_batch_size`가 숫자이면 L-MPDD tile, anchor plane, final plane,
VAE Refine을 나눠 peak memory를 제한한다. `null`이면 각 단계의 작업을 한 batch로
처리한다. 배치 방식은 메모리만 바꾸며 decoder interpolation과 후보 평가 의미는 같다.
3D component,
percolation, Euler density는 최종 진단으로 기록하지만 단일 2D reference에서 정답을
추론할 수 없으므로 아직 gate로 사용하지 않는다.

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
scale slice schedule, scalar phase optimizer, 순차 slice overwrite도 삭제했다.

## 검증

예측의 L-MPDD sampling, Joint guidance, scale-up sampling/guidance는 `progress`
설정을 공유하고 노트북에서는 기본적으로 진행률을 일반 텍스트 `tqdm`으로 표시한다.
GAN 학습은 generator/critic loss, margin, gradient penalty, fraction error를 표시하고,
Joint는 전체 loss와 활성화된 anchor·critic·global fraction loss를 갱신한다.

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
- GAN 학습 중 VAE weight는 고정된다.
- frozen critic guidance는 residual parameter에 finite gradient를 전달한다.
- phase fraction의 global 조건은 기본적으로 개별 slice fraction을 강제하지 않는다.
- quality gate를 통과하지 못해도 least-violation 후보와 실패 통계를 반환한다.

회귀 검증:

- categorical VAE, L-MPDD, scale-up, 다중 축 앵커, phase fraction API를 보존한다.
- `.venv\\Scripts\\python.exe -m pytest -q` 전체 테스트를 통과한다.
- `04_predict.ipynb`는 통합 기본 크기 예제로, `05_scale_up.ipynb`는 같은 상위
  파이프라인의 scale 예제로 실행 가능해야 한다.
