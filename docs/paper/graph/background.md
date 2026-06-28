# 2D 이미지에서 3D 후보를 만드는 이유

이 논문의 출발점은 제한된 영역을 담은 2D 이미지다. 논문은 3D 정답 데이터가 많이 있다고 가정하지 않는다. [1](#ref-1)

## 2D 조각 이미지의 장점

- 2D SEM 관찰은 같은 시설 요금표 안에서 3D FIB-SEM보다 시간당 비용이 낮다.

  기관 소속 사용자는 그 기관 안 연구자이고, 기관 외부 사용자는 기관 밖 연구자다. 비교는 같은 사용자 구분끼리 보면 된다. [3](#ref-3)

  | 근거 | SEM 기관 소속 | FIB-SEM 기관 소속 | SEM 기관 외부 | FIB-SEM 기관 외부 | FIB-SEM/SEM |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | GW | $35/hour | $80/hour | $105/hour | $240/hour | 2.3x |

  같은 GW 요금표 기준으로 FIB-SEM은 SEM보다 시간당 약 2.3배 비싸다. UC Merced의 SEM 요금도 기관 소속 $27.50-$33.09/hour, 기관 외부 $42.63-$51.27/hour 구간이다. [2](#ref-2) [3](#ref-3)
- 시간 면에서도 2D 사진 한 장을 확인하는 일과 3D 부피 전체를 촬영하는 일은 범위가 다르다.

  ORNL의 탁상형 SEM 설명은 시료를 30초 안에 넣을 수 있고, 10Hz로 화면을 갱신한다고 적는다. 모든 SEM 실험이 이 속도라는 뜻은 아니다. 여기서 중요한 차이는 작업 단위다. 2D는 한 화면을 확인하는 일이고, 3D는 각도나 층을 바꿔 여러 장을 모아 부피를 만드는 일이다. [4](#ref-4) [8](#ref-8) [9](#ref-9)
- 논문도 정답 없는 2D 조각 이미지에서 출발한다.

  논문은 2D 현미경 이미지 하나를 잘라 만든, 정답 표시가 없는 2D 조각 이미지들을 출발점으로 삼는다고 쓴다. 처음부터 3D 정답 데이터가 많이 있다고 가정하지 않는다. [1](#ref-1)

## 단면 사진의 한계

알고 싶은 것은 재료 속 연결이다. 빈 공간이나 각 물질이 어떻게 이어지는지를 알아야 기체나 이온의 길, 막힌 길, 전체 성능 변화를 판단한다. [1](#ref-1)

단면 한 장은 이 연결을 확정하지 못한다. 끊겨 보이는 길이 실제 입체 구조에서는 이어질 때가 있고, 이어진 것처럼 보이는 길이 실제로는 막혀 있을 때도 있다. [1](#ref-1)

```text
가진 것: 재료 단면 이미지
필요한 것: 재료 속 입체 구조와 그 구조가 만드는 성능
어려운 점: 속을 직접 많이 찍기 어렵고, 단면만으로는 내부 연결을 알 수 없다
```

## 3D 촬영의 부담

- 아래 사례에서 3D 장비 도입 비용은 백만 달러 이상으로 올라간다.

  나노미터 수준 3D X-ray 현미경은 170만 달러짜리 장비 사례가 있다. FIB-SEM 도입에는 250만 달러 규모의 지원금이 붙은 사례가 있다. 더 특수한 cryo plasma-FIB SEM은 500만 유로가 넘는 장비로 소개된다. [5](#ref-5) [6](#ref-6) [7](#ref-7)
- 3D 촬영 시간은 30-60분에서 며칠까지 나온다.

  CT 시설 안내는 일반 스캔을 보통 30-60분으로 적는다. 장비별 범위로는 20초-3시간 이상, 15분-20시간 이상도 제시한다. [8](#ref-8)

  FIB-SEM 3D 작업은 이미지만 12-48시간, 분석까지 넣으면 3-4일로 소개된다. [9](#ref-9)
- 3D 장비는 사용료도 누적된다.

  | 근거 | 장비 | 기관 소속 | 기관 외부 |
  | --- | --- | ---: | ---: |
  | Cornell [10](#ref-10) | micro-CT Skyscan 1276 | $110/hour | $180-$275/hour |
  | BU [11](#ref-11) | Zeiss Xradia Versa 520 | $91.15/hour | $162.25/hour |
  | ASU [12](#ref-12) | SEM/FIB Helios 5 UX | $76.44/hour | $162.96/hour |

  이런 장비는 시간당 비용에 촬영 시간이 곱해진다. 그래서 12-48시간짜리 FIB-SEM 촬영은 비용 부담도 같이 커진다. [9](#ref-9) [10](#ref-10) [11](#ref-11) [12](#ref-12)
- 일부 3D 방식은 같은 부피를 다시 보기 어렵다.

  2D 단면 준비에도 절단이나 연마가 들어갈 때가 있다. 차이는 FIB-SEM 같은 3D 방식에서는 보고 싶은 부피를 얇은 층으로 계속 깎아가며 찍는다는 점이다. 깎아낸 부분은 되돌릴 수 없다. [13](#ref-13)

## 논문이 다루는 문제

이 논문은 단면 이미지에서 3D 후보를 만드는 방법을 제안한다. 여기서 멈추면 후보를 원하는 방향으로 바꾸지는 못한다. 설계에서는 “이 물질을 더 많게 만들고 싶다”, “서로 닿는 면을 늘리고 싶다”, “길이 더 잘 이어지게 만들고 싶다” 같은 목표 수치가 생긴다.

이 논문은 3D 후보를 만드는 것뿐 아니라, 원하는 방향으로 후보를 바꾸는 방법까지 다룬다. [1](#ref-1)

이 배경에서 나오는 핵심 주장이 [[docs/paper/graph/claim|논문의 주장]]이다. MicroLad는 2D 조각 이미지에서 3D 후보를 만들고, 그 후보를 목표 수치에 맞게 조정한다고 주장한다.

## References

1. <a id="ref-1"></a>[MicroLad 원문 추출본](../md/2508.20138v4.md)
2. <a id="ref-2"></a>[UC Merced Imaging and Microscopy Facility rates](https://imf.ucmerced.edu/rates)
3. <a id="ref-3"></a>[GW Nanofabrication and Imaging Center usage fees](https://nic.gwu.edu/usage-fees)
4. <a id="ref-4"></a>[ORNL Phenom XL SEM](https://www.ornl.gov/content/phenom-xl-sem)
5. <a id="ref-5"></a>[WSU JCDREAM Xradia 810 Ultra](https://jcdream.org/xray-computed/)
6. <a id="ref-6"></a>[WVU FIB-SEM](https://sharedresearchfacilities.wvu.edu/news/2024/07/02/new-focused-ion-beam-scanning-electron-microscope-for-the-wvu-shared-research-facilities)
7. <a id="ref-7"></a>[Goethe cryo plasma-FIB SEM](https://www.uni-frankfurt.de/en/newsroom/meldungen/pressemitteilungen/2026/blick-ins-innerste-des-lebens-erstes-rasterelektronenmikroskop-mit-nanomanipulator-in-hessen-an-der-goethe-universitaet-eingeweiht)
8. <a id="ref-8"></a>[UTCT Scanning FAQ](https://www.ctlab.geo.utexas.edu/scanning-faq/)
9. <a id="ref-9"></a>[Thermo Fisher FIB serial sectioning Q&A](https://documents.thermofisher.com/TFS-Assets/MSD/Reference-Materials/qa-report-unattended-fib-serial-sectioning-tomography.pdf)
10. <a id="ref-10"></a>[Cornell X-ray micro-CT pricing](https://www.biotech.cornell.edu/core-facilities-brc/services/x-ray-micro-ct)
11. <a id="ref-11"></a>[BU Micro-CT rates](https://www.bu.edu/odbl/rates/)
12. <a id="ref-12"></a>[ASU Materials Core rates](https://cores.research.asu.edu/materials/rates)
13. <a id="ref-13"></a>[Plymouth Serial Sectioning Tomography](https://www.plymouth.ac.uk/facilities/plymouth-electron-microscopy-centre/techniques/serial-sectioning-tomography)
