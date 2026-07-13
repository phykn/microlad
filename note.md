# 3D 조건부 복원 실험 노트

마지막 갱신: 2026-07-13

## 목표와 판정 순서

1. 지정한 위치와 축의 하나 이상 단면이 조건 이미지와 **유사**해야 한다. 조건 이미지를 그대로 복사하지는 않는다.
2. XY뿐 아니라 XZ/YZ 단면도 조건 이미지와 비슷한 2D 형태를 가져야 한다.
3. 인접 단면은 연속적으로 변해야 하며, 앵커에서 3칸을 벗어난 지점에서 갑자기 바뀌면 실패다.
4. 같은 XY가 z 방향으로 반복되어 XZ/YZ에 긴 기둥이나 띠가 생기면 실패다.
5. 시각적 형태를 먼저 보고, 그다음 앵커 오차·축 전이율·상분율을 확인한다. 수치만 좋아지고 파편이나 잡음이 늘면 채택하지 않는다.

초기 고정 진단 조건 이미지의 참고값:

- 상분율: `[0.3540, 0.1062, 0.5398]`
- 2D 이웃 전이율: `0.2228`

고정 seed를 완료 조건으로 사용하지 않으므로 위 값은 과거 sweep 비교용일 뿐이다. 실제 실행에서는 그 실행이 선택한 조건 이미지의 상분율과 형태를 다시 계산해 판정한다.

용어: 공극률은 별도 조건이 아니라 공극 phase의 `phase_fractions` 값이다. 모든 phase fraction을 합이 1이 되도록 함께 입력하며 기본 허용오차는 ±1%p다.

## 현재 채택한 기반

- categorical VAE의 확률 출력을 사용한다.
- 2D 한 장을 세 축에 독립적으로 복원하지 않고, 하나의 공유 3D generator가 만든 volume의 XY/XZ/YZ **모든 64개 단면**을 같은 2D critic으로 학습한다.
- 기본 texture 학습에는 조건 이미지의 회전·반사·주기 이동본을 사용한다. categorical VAE와 2D diffusion이 만든 8개 단면은 조건 이미지와 같은 상분율로 보정한 뒤 `10%`만 섞는다.
- texture 학습은 충분히 성숙한 primary `5000` step과 hybrid `+500/+1000` step만 조건화 후보로 사용한다. `3000/4000` 후보는 수치가 좋아도 길고 거친 domain을 선택하는 경우가 있어 제외했다.
- 각 후보에서 먼저 `4³` spatial noise를 조건 이미지에 맞추고, 그 다음 critic을 고정한 채 generator와 noise를 짧게 미세조정한다.
- 앵커를 voxel에 직접 복사하거나 ±몇 칸의 slab로 고정하지 않는다. 조건은 generator의 공간장 전체를 통해 전달되므로 ±3 밖에 별도의 경계가 없다.
- 실행마다 정확히 같은 volume을 다시 만드는 것은 목표가 아니다. 매 실행 결과 자체가 앵커 유사도·세 축 morphology·국소 연속성 기준을 통과하는지를 판정한다.

현재 채택한 다중 앵커·상분율 실행 결과(`03_predict.ipynb`):

- 앵커: 중심 XY와 XZ, 교차선이 일치하는 categorical 이미지 두 장
- 선택 상태: hybrid `6000` step, 조건 후보 3개 비교
- 앵커 mismatch: `[8.08%, 7.98%]` — 두 앵커 모두 유사하지만 복사하지 않음
- 목표/실제 상분율: `[0.28, 0.12, 0.60]` / `[0.28, 0.12, 0.60]`
- 축 전이율: `[0.2207, 0.2226, 0.2094]`, 축간 spread `0.0132`
- 앵커 주변 최대 boundary jump: XY축 `0.0352`, XZ축 `0.0425`
- 실행 시간: RTX 2060에서 `758.2초`
- 시각 판정: 두 앵커가 복사되지 않았고 ±6 단면이 점진적으로 변한다. 전역 XY/XZ/YZ 모두 둥근 blob 형태를 유지하며 hard cutoff가 없다.

현재 가장 좋은 진단 결과(저장된 hybrid 6000-step 상태):

- 앵커 mismatch: `7.37%` — 조건과 유사하지만 동일하지 않음
- 상분율: `[0.3583, 0.1019, 0.5399]`
- 축 전이율: `[0.1867, 0.1883, 0.1832]`
- lag-3: `[0.3876, 0.3932, 0.3859]`
- 축별 run-profile MAE: `[0.0277, 0.0325, 0.0296]`
- 앵커 주변 z-boundary profile: `[0.2007, 0.1990, 0.1877, 0.2070, 0.1667, 0.1880, 0.1682, 0.1738, 0.1816, 0.2063]`
- 시각 판정: z=26..38이 점진적으로 변하고, ±3 밖 cutoff가 없으며, 세 축 모두 조건과 비슷한 blob morphology를 보임
- 남은 차이: 회색 phase의 2D Euler 수가 조건 약 `66`보다 생성 약 `40~41`로 낮아 작은 독립 domain은 아직 부족함

## 실험 기록

| 실험 | 주요 설정/결과 | 판정 | 이유 |
|---|---|---|---|
| 원형에 가까운 SliceGAN, anchor refs | 4³ noise, 모든 64개 단면/축, WGAN-GP, 5000 step | 채택 | 3000~5000 step에서 세 축의 blob morphology와 연속성이 처음으로 함께 안정됨 |
| 다중 축 앵커 + 명시적 상분율 | 중심 XY/XZ 두 장, fraction `[0.28, 0.12, 0.60]`, 교차선 사전 검증 | 채택 | mismatch `[8.08%, 7.98%]`, 상분율 정확, 축 전이율 spread `0.0132`; 두 조건과 세 축 자연스러움을 동시에 통과 |
| 기존 dynamic-interpolation generator를 큰 noise grid에 적용 | 256³ full/tiled 직접 비교 | scale-up에서 폐기 | 마지막 보간 배율이 입력 크기에 따라 달라져 voxel morphology scale이 유지되지 않음. tiled probability 차이는 작지만 near-tie categorical label이 `30.04%` 달라짐 |
| fully-convolutional scale generator | 4회 `k=4,s=2,p=1` upsampling과 local `3³` 출력, 64³ 초과에만 적용 | 코드 채택 | `4→64`, `8→128`, `16→256`이 같은 voxel scale을 유지하고 halo tiled 렌더가 full 렌더와 일치함. 64³ 채택 경로는 변경하지 않음 |
| 앵커 주변 목표 전이율 loss | 앵커 법선 방향의 전이를 0으로 만들지 않고 학습 texture의 전이율에 맞춤 | 채택 | 앵커 평면 밖에서 갑자기 형태가 바뀌는 cutoff를 완화함 |
| Gaussian 원본장 보존 | 앵커 패치에서 멀수록 조건화 전 generator 확률장을 보존 | 채택 | hard slab 없이 앵커 영향의 공간 범위를 부드럽게 제한함 |
| 조기 4000-step 조건 후보 | 목표 mismatch 9%, 수치 gate 통과 | 폐기 | 전역 단면이 길고 거친 domain으로 변해 시각 우선 기준을 통과하지 못함 |
| diffusion reference 50% 직접 학습 | categorical VAE+2D diffusion 단면 8개 | 폐기 | critic margin 약 `99`, 상 소실과 파편·띠 구조가 발생함 |
| anchor 90% + diffusion reference 10% hybrid | 좋은 anchor checkpoint에서 1000 step 추가 | 채택 | step 6000 상분율 `[0.3580, 0.1057, 0.5363]`, margin 약 `29`; anchor-only 형태를 보존하면서 diffusion prior를 약하게 반영함 |
| frozen generator에서 noise만 800 step 조건화 | baseline mismatch 약 `29%`, hybrid 약 `25%` | 중간 단계만 채택 | 3D 연속성은 보존하지만 지정 단면 유사도가 부족함 |
| frozen critic, generator+noise 조건 미세조정 | G lr `1e-5`, noise lr `2e-3`, critic `0.02`, phase `50`, 최대 500 step | 채택 | mismatch `7.37%`, 변경 voxel `12.1%`; 앵커를 복사하지 않고 전역 구조도 유지함 |
| 좋은 SliceGAN 결과에 기존 joint 후처리 | anchor mismatch 약 `8%` | 폐기 | z=35→36에 새 cutoff가 생겨 원래의 연속성을 손상함 |
| 조건 미세조정에 continuity weight `0.01` | 나머지는 채택 설정과 동일 | 폐기 | 조건 형태와 상분율 손실에 비해 경계 profile 개선이 없었음 |
| morphology 기반 multi-checkpoint 선택 | primary 3000/4000/5000, hybrid +500/+1000; 상분율·전이율·run profile + 조건·경계 판정 | 채택 | 다른 조건 crop에서도 단일 final checkpoint의 GAN 변동성을 줄이고 내부 품질 기준을 통과함 |
| hard anchor slab | 앵커 주변 반경 2를 직접 고정 | 폐기 | z=29/35 부근에서 고정 영역 밖 구조가 갑자기 깨짐 |
| Gaussian anchor latent spread | 중심 latent를 주변 z에 확산 | 폐기 | 비슷한 XY가 반복되어 XZ/YZ에 긴 기둥과 띠가 생김 |
| tri-plane consensus + center-only soft anchor | 위의 안정적 기준 결과 | 채택 | hard cutoff와 강한 축 비대칭을 가장 안정적으로 줄임 |
| transpose-conv online SliceGAN 형태 | 300 step | 폐기 | grid/checkerboard collapse, 앵커 mismatch 약 51% |
| Patch/WGAN guidance | 짧은 online 학습 | 기본값에서 비활성 | discriminator가 generator를 압도하고 형태가 불안정 |
| SWD texture guidance | multi-scale patch distribution | 폐기 | 큰 덩어리로 과도하게 평활화됨 |
| phase-interface guidance | phase pair 경계 통계 | 폐기 | 큰 domain을 만들고 조건 형태와 멀어짐 |
| categorical VAE 1회 3축 투영 | mismatch `13.96%`, 전이율 `[0.1677, 0.1751, 0.1786]` | 폐기 | 앵커와 상분율이 악화되고 XZ/YZ 큰 덩어리가 커짐 |
| 저노이즈 diffusion 후처리, t=25 | 변경량 1.5~9.8% sweep | 폐기 | 모양은 거의 그대로이고 강할수록 경계만 평활화됨 |
| 저노이즈 diffusion 후처리, t=100 | 변경량 2.0~10.9% sweep | 폐기 | XZ/YZ의 긴 구조가 줄지 않고 앵커 오차만 증가 |
| global transition weight=1.0 | 전이율 `[0.2097, 0.2139, 0.2145]`, mismatch `10.60%` | 폐기 | 목표 전이율에는 가까워졌지만 잔경계와 파편이 과도함 |
| transition=0.5, continuity=0.002, SA=1.0 | 전이율 `[0.1956, 0.2039, 0.2044]`, mismatch `10.18%` | 폐기 | 긴 덩어리는 줄었지만 XZ/YZ 파편화가 여전히 큼 |
| 낮은 SDS timestep 범위 | `t=10..200`, transition `0.35`, continuity `0.002`, SA `1.0`, anchor `0.075` | 이전 기준 | mismatch `8.69%`, 전이율 `[0.1913, 0.2031, 0.2051]`, 상분율 `[0.3521, 0.1005, 0.5474]`; transition=1의 파편화를 완화했지만 XZ/YZ가 XY만큼 둥글지 않음 |
| multi-scale run profile | transition을 끄고 길이 `2/4/8/16` run loss `0.25` | 현재 채택 | mismatch `7.01%`, 전이율 `[0.1769, 0.1882, 0.1865]`, 상분율 `[0.3589, 0.0976, 0.5434]`, 축별 run MAE `[0.0372, 0.0271, 0.0291]`; 파편과 앵커 오차가 줄었으나 XZ/YZ의 연결 topology는 조건과 여전히 다름 |
| phase별 Euler density | run profile에 Euler loss `1e-5` 추가 | 폐기 | mismatch `8.03%`, 상분율 `[0.3574, 0.0861, 0.5565]`; 회색 phase 목표 `66.4` 대비 세 축 약 `33`이고 시각적 연결망도 개선되지 않아 단일 topology 스칼라가 불충분함 |
| 직접 64³ voxel logits | Conv3D 보정장을 없애고 각 voxel을 직접 최적화, lr `0.02` | 폐기 | mismatch `0.00%`로 앵커를 그대로 복사했고, 중심 경계가 `0.5247`로 주변 중앙값 `0.2651`보다 급증했다. 전이율 `[0.2651, 0.2653, 0.2656]`, run MAE `[0.0572, 0.0519, 0.0546]`; 세 축 모두 심하게 파편화되어 3D 공간 정규화가 반드시 필요함 |
| 매 step 세 축 동시 slice batch | 총 batch 16을 XY/XZ/YZ에 나누어 매 step 동시에 최적화 | 폐기 | mismatch `7.25%`, 전이율 `[0.1737, 0.1844, 0.1850]`, run MAE `[0.0367, 0.0287, 0.0298]`; 기존 run-profile 결과와 시각적으로 거의 같아 축을 번갈아 갱신한 순서가 주원인은 아님 |
| 안정화 categorical slice critic, weight=0.02 | hard categorical 입력, spectral normalization, hinge loss, 60-step ramp | 보류 | 상분율 `[0.3553, 0.1024, 0.5422]`, mismatch `8.96%`, 전이율 `[0.1833, 0.1957, 0.1908]`; 회색 phase Euler가 약 `29~31`에서 `33~35`로 개선됐지만 critic 항이 평균 loss의 약 2%라 XZ/YZ 연결망 변화는 작음 |
| 안정화 categorical slice critic, weight=0.10 | 나머지는 weight `0.02`와 동일 | 폐기 | mismatch `14.53%`, 상분율 `[0.3266, 0.1139, 0.5594]`, run MAE `[0.0404, 0.0436, 0.0426]`; critic margin은 `0.27→0.25`로 거의 줄지 않은 채 다른 조건만 무너져 단순 가중치 증가는 해법이 아님 |
| 초기 L-MPDD logit strength=0.25 | critic `0.02`, 나머지는 동일 | 폐기 | mismatch `11.77%`, 전이율 `[0.1595, 0.1702, 0.1676]`, run MAE `[0.0497, 0.0437, 0.0442]`; 큰 매끈한 연결망으로 수렴하고 critic margin이 `0.94`로 커져 초기장 결합만 약화하는 것은 해법이 아님 |
| local categorical slice critic, RF=22, weight=0.02 | 46px critic에서 receptive field만 22px로 축소 | 보류 | mismatch `8.57%`, 상분율 `[0.3583, 0.1021, 0.5396]`, run MAE `[0.0360, 0.0278, 0.0285]`, margin `0.19`; 수치 안정성과 회색 Euler `34~36`은 개선됐지만 XZ/YZ 시각 변화는 아직 작음 |
| local categorical slice critic, RF=22, weight=0.05 | 나머지는 RF=22, weight `0.02`와 동일 | 폐기 | mismatch `10.40%`, 상분율 `[0.3388, 0.1118, 0.5495]`, run MAE `[0.0395, 0.0357, 0.0336]`; 회색 Euler는 `36~37`로 올랐지만 조건·상분율 손실과 XZ/YZ 연결망을 함께 해결하지 못함 |
| multiscale logit pyramid | `8³+16³+32³`, lr `0.01`, local critic `0.02` | 폐기 | mismatch `12.33%`, 상분율 `[0.3355, 0.0988, 0.5658]`, run MAE `[0.0384, 0.0328, 0.0364]`; 작은 독립 domain은 늘었지만 조건 손실과 critic margin `0.73`이 커지고 전체 형태가 여전히 다름 |
| fixed-noise online generator, 600 step | 4³ noise, uniform-information generator, 전체 192 slice critic, anchor `0.25` | 폐기 | mismatch `8.50%`지만 상분율 `[0.3821, 0.0563, 0.5616]`, run MAE `[0.0545, 0.0375, 0.0939]`; XZ/YZ에 규칙적인 대각선·격자 패턴이 생겨 한 latent에 texture와 anchor를 동시에 강제하는 방식은 실패 |
| 2-stage online generator, 1200 step | random-noise texture `900` + frozen-generator latent condition `300` | 폐기 | 회색 phase가 `0%`로 소실되고 mismatch `54.05%`, critic margin `3.11`; generator가 critic에 완전히 압도되어 모든 축에 강한 주기 패턴이 생김 |
| image-space harmonization, anchor=0.15 | 4 sweep, t=`100`, blend=`0.25`, coarse latent anchor 미사용 | 폐기 | 전체 변화량 `9.18%`, mismatch `1.76%`로 앵커를 사실상 복사했고 run MAE가 약 `0.041~0.047`로 악화됨; 회색 Euler `38~39` 개선만으로는 채택 불가 |
| image-space harmonization, anchor=0.02 | 나머지는 anchor=`0.15`와 동일 | 폐기 | mismatch `9.35%`로 복사 문제는 해결했지만 전체 변화량 `8.99%`, run MAE 약 `0.042~0.047`; XZ/YZ morphology가 거의 개선되지 않아 반복 저노이즈 projection도 해법이 아님 |

현재 `03_predict.ipynb`와 `04_scale_up.ipynb`는 `config/slicegan.yaml`을 읽는다. 사용자 코드는 `PredictOptions(..., slicegan=slicegan_config)`만 지정하며, 세부값은 다음 세 묶음으로 관리한다.

```text
training: GAN 학습과 diffusion reference 혼합
conditioning: 앵커 조건화와 generator/noise 미세조정
rendering: scale-up 타일 크기와 halo
```

구형 flat 인자 호환 계층은 사용처가 없어 제거했다. 네트워크는 `src/modeling/slicegan.py`, 앵커 모델·교차 검증은 기존 `guidance/conditioning`, scale-up 타일 렌더는 기존 `pipelines/scaling`, 실행 조율만 `guidance/slicegan.py`가 담당한다.

## SliceGAN 원 구현에서 확인한 차이

- 원 논문은 한 생성 volume에서 XY/XZ/YZ의 **64개 단면 전부**를 critic에 전달한다. 최소 32개, 실제로는 64개 모두가 더 안정적이라고 보고한다.
- 공식 구현은 `4³` spatial noise, uniform-information transpose convolution generator, WGAN-GP, generator/discriminator lr `1e-4`, critic iteration `5`, 100 epochs를 사용한다.
- 논문에 보고된 학습 시간은 Titan Xp 기준 약 4시간이다. 따라서 앞선 300-step, 16-slice 보조 critic은 SliceGAN을 충분히 재현한 실험이 아니며, "짧은 GAN"의 한계로 해석한다.
- 참고: [SliceGAN 논문](https://arxiv.org/abs/2102.07708), [공식 구현](https://github.com/stke9/SliceGAN)

## 해석

- XZ에서 구조가 길어지는 직접 원인은 z 방향으로 같은 상 배치가 오래 유지되기 때문이다. 거의 같은 XY가 반복되면 XZ/YZ에서는 세로 기둥이나 긴 띠가 된다.
- 2D diffusion prior만으로도 원칙상 가능하지만, 세 축의 2D 조건을 **한 공유 3D voxel field에서 동시에 만족**시켜야 한다. 축별로 독립 생성한 뒤 합치는 방식은 이 일관성을 보장하지 못한다.
- 전이율 하나만 맞추면 모델은 작은 파편을 추가하는 쉬운 해를 선택한다. 따라서 전이율 개선은 시각적 domain 크기·형태와 함께 판단해야 한다.
- run profile은 연속 길이와 파편 문제를 개선하지만, 같은 길이 분포라도 분리된 둥근 영역과 서로 연결된 망은 구분하지 못한다.
- 기존 joint-SDS 경로의 병목은 `phase별 2D topology(연결 성분과 hole의 균형)`였고, SliceGAN의 공유 3D generator가 이를 크게 개선했다.
- 2D diffusion reference를 많이 섞으면 오히려 학습이 무너진다. 현재 diffusion은 새로운 texture 전체를 결정하는 teacher가 아니라 categorical 형태 다양성을 약하게 보완하는 prior로 사용한다.
- 조건 이미지는 한 voxel 면에 붙여 넣는 대상이 아니라, generator의 spatial noise와 weight가 만들어 내도록 유도하는 목표다. 그래서 중심 단면을 완전히 같게 만드는 것보다 `약 7~8% mismatch`를 허용하는 편이 조건성과 3D 자연스러움의 균형이 좋다.

## 다음 진행 규칙

- 한 번에 가설 하나만 바꾼다.
- 정확한 seed 재현은 요구하지 않는다. 비교가 필요한 단일 sweep 안에서는 같은 입력을 유지하되, 최종 판정은 각 실행 결과의 품질 기준으로 한다.
- 각 실행 뒤 이 표에 파라미터, 핵심 수치, 시각 판정, 채택 여부를 기록한다.
- 폐기한 방법은 새로운 근거가 없으면 다시 켜지 않는다.
- `04_scale_up.ipynb` 정식 128³ 실행과 전 축 시각 검증까지 완료했다. 이후에는 저장 artifact 재검토 또는 새로운 조건 실험에 사용한다.

## 요구사항 완료 감사

| 요구사항 | 현재 증거 | 상태 |
|---|---|---|
| 64³ 복수·다축 앵커, 비복사, 교차 충돌 | `03_predict.ipynb` 전체 실행과 전 축 몽타주, mismatch `[8.08%, 7.98%]`, 충돌/중복 테스트 | 완료 |
| 사용자 phase fraction | 64³ 전체 실행 및 128³ smoke에서 목표 fraction 정확, validation/regression 테스트 | 완료 |
| 전역 세 축 자연스러움 | 64³ 전 축 몽타주와 국소 ±6 시각 판정 통과, 공통 run/Euler/repetition/cutoff QA 추가 | 64³ 완료 |
| 절대 좌표의 같은 축·다른 축 복수 앵커 | 128³ 실제 Predictor smoke와 offset 교차선/patch 테스트 | 코드 경로 완료 |
| 최소 128³ 품질 | `04_scale_up.ipynb` 정식 6000-step 실행, 수치 gate 6개와 세 축 128개 전 단면 시각 검증 통과 | 완료 |
| 임의 배수 및 bounded/tiled memory | 64 배수 fully-convolutional noise-grid API, 256³ full/tiled 최대 오차 `5.96e-8`, GPU peak `5.15→2.34 GiB`; 큰 training forward는 activation checkpoint 적용 | 최종 렌더 완료, 대형 조건 최적화는 checkpointed full-grid이며 완전 tile 방식은 미완료 |
| 시간 분리 기록 | reference/training/generation/total 통계 구현 및 smoke 실측 | 완료 |

## 검증 상태

- `03_predict.ipynb`: 다중 축 앵커와 명시적 상분율 경로로 처음부터 끝까지 실행 완료 (`758.2초`, CUDA)
- 최종 노트북 실행: step `6000` 선택, 조건 후보 `3개` 비교, 앵커 mismatch `[8.08%, 7.98%]`
- 최종 노트북 실행 상분율: 조건과 전체 모두 `[0.28, 0.12, 0.60]`
- 최종 노트북 실행 축 전이율: `[0.2207, 0.2226, 0.2094]`
- 최종 노트북 실행 앵커 주변 최대 인접 jump: `[0.0352, 0.0425]`
- 최종 시각 판정: 중심 조건 두 장을 복사하지 않고 큰 domain 배치를 유지함. 각 앵커 ±6은 누적해서 변하며 ±3 밖에 hard cutoff가 없음. XY/XZ/YZ 전역 몽타주는 모두 blob morphology를 유지함
- 노트북 JSON, cell id, 모든 코드 셀 문법 검증 완료
- 실제 Predictor 1-step multi-anchor/fraction smoke: `64³`, categorical `uint8`, 목표 상분율 정확, OOM 없이 통과
- 실제 Predictor 128³ fully-convolutional 1-step 통합 smoke: 같은 축 절대 index `20/100`과 다른 축 index `64`를 동시에 전달해 `(128,128,128)` categorical `uint8` 출력, fraction `[0.28,0.12,0.60]` 정확, tolerance 및 전역 QA 통과, peak GPU memory `1.805 GiB`
- phase fraction 허용오차 smoke: 공극률을 포함하는 `phase_fractions=(0.28,0.12,0.60)`의 128³ 실측 절대 오차 `[2.1e-7,1.1e-7,1.2e-7]`, 기본 tolerance `0.01` 통과. 공극률은 별도 입력이 아니라 지정한 공극 phase의 fraction이다.
- 공통 전역 QA: 세 축 transition/lag-3, 완전 반복 단면 비율, 전체 최대 boundary jump, 축별 run-profile MAE, 축별 Euler topology MAE, fraction error를 모든 SliceGAN 결과 통계에 기록한다. 실제 128³ 통합 smoke에서 모든 항목의 shape와 finite 값을 확인함
- tiled inference: 64³ 초과에서는 local fully-convolutional generator를 사용하고 latent core 4칸에 halo 4칸을 붙여 core만 조립한다. 256³ full/tiled 직접 비교에서 probability MAE `3.91e-11`, 최대 `5.96e-8`, label 차이 `2.38e-7`; full `2.86초/5.150 GiB`, tiled CPU-output `19.53초/2.338 GiB`. 이는 최종 inference activation을 제한하며 대형 조건 최적화 전체를 tile화한 것은 아니다.
- 안전장치: global interpolation을 사용하는 기존 64³ generator는 tiled renderer에 전달하면 명시적으로 거부한다. `04_scale_up.ipynb`는 fully-convolutional 상태와 fraction tolerance를 출력·검증한다.
- 대형 조건 최적화: noise grid가 8보다 크면 generator forward에 non-reentrant activation checkpoint를 사용한다. 출력과 noise/parameter gradient가 direct forward와 일치하는 회귀 테스트를 추가했고, 192³ noise-gradient smoke는 `2.72초`, peak `4.975 GiB`, finite gradient로 통과했다. 출력 tensor 자체가 커지므로 이것만으로 256³ finetune을 보장하지는 않는다.
- 128³ 앵커 좌표: `AnchorSlice.index`를 출력 volume의 절대 좌표로 보존하고 64×64 이미지만 해당 평면 중앙에 배치한다. 같은 축의 복수 절대 index와 서로 다른 축의 offset 교차선 검증을 테스트로 고정함
- 실행 시간 통계: 다음 실행부터 reference 준비, texture 학습, 앵커 조건 생성, 전체 시간을 각각 `slicegan_*_seconds`로 기록함
- `04_scale_up.ipynb` 정식 실행: CUDA `1601.1초` (`26분 41초`), reference `15.4초`, texture/hybrid 학습 `663.1초`, 128³ 조건 생성 `922.4초`, 선택 step `6000`
- 정식 128³ 앵커 mismatch `[7.50%, 7.45%]`; 두 앵커 모두 유사하지만 hard copy하지 않음. 목표/실측 fraction 모두 `[0.28,0.12,0.60]`
- 정식 128³ 축 전이율 `[0.2554,0.2490,0.2547]`, run-profile MAE `[0.0096,0.0085,0.0114]`, Euler MAE `[2.4764,4.3670,2.3823]`, lag-3 `[0.4012,0.4087,0.4004]`
- 정식 128³ 완전 반복 단면 비율 `[0,0,0]`, 전체 최대 boundary jump `[0.0287,0.0396,0.0251]`; 앵커/fraction/축 spread/반복/cutoff/run gate 6개 전부 통과
- 정식 128³ 시각 판정: 동일 64×64 center crop에서 조건과 같은 큰 blob 크기와 회색 미세상을 유지한다. 앵커 ±8 단면은 두 축 모두 점진적으로 변하고 ±3 밖 cutoff나 slab가 없다. XY/XZ/YZ 각각 128개 전 단면 contact sheet에서 긴 기둥, checkerboard, tile 반복, 축별 collapse, phase-mixing 파편이 보이지 않아 채택
- 정식 실행 종료 시 widget kernel shutdown의 `KeyboardInterrupt caught in kernel` 로그가 있었으나 nbconvert exit `0`, 실행 코드 셀 `5/5`, cell error `0`, NPZ 저장·재로딩 검증 통과
- `04_scale_up.ipynb` 임시 1-step end-to-end smoke: 5개 코드 셀 모두 실행, 오류 0, CUDA `21.5초`; `(128,128,128)` categorical 출력, 절대 앵커 `[(0,64),(1,64)]`, fully-convolutional=True, fraction 정확, 전역 QA·몽타주·슬라이더까지 통과. mismatch `[50.73%,44.63%]` 등은 1-step 미학습 결과라 품질 증거로 사용하지 않음
- 04 전역 보조 gate: 완전 반복 인접 단면 `0`, 전체 최대 boundary jump `≤0.08`, 세 축 run-profile MAE `≤0.05`를 자동 판정하고 세 축 전체 boundary profile을 그린다. Euler topology는 단일 스칼라만으로 합격을 결정하지 않고 비교 지표로 유지한다.
- `04_scale_up.ipynb` 장시간 실행 결과는 `run/20260712-163751-714469/predictions/conditional_slicegan_128.npz`에 categorical volume, 모든 tensor 통계, 앵커 이미지/축/절대 index, phase fractions, 주요 step 설정을 함께 자동 저장하고 즉시 재로딩 검증하도록 준비함
- 이후 `USE_SAVED_RESULT=True`로 설정하면 generator를 다시 학습하지 않고 artifact를 불러와 QA·몽타주·슬라이더만 재실행한다. 현재 앵커 이미지/축/index, fraction과 fraction/intersection tolerance, step 설정, volume shape가 저장 조건과 다르면 명시적으로 거부함
- 저장 checkpoint 기반 전체 조건화/시각 검증: 통과
- multi-checkpoint 통합 Predictor 전체 실행: 내부 mismatch `8.03%`, phase MAE `0.00164`, 경계 표준편차 `0.02882`, 최대 국소 경계 jump `0.05029`; 내부 기준 통과
- focused tests: `66 passed, 33 subtests passed`
- 전체 테스트: 변경 후 `469 passed, 106 subtests passed`
